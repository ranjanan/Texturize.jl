using Images
# using ProgressMeter
using MXNet

function preprocess_content_image(img, longEdge::Int)
    println("load the content image, size = $(img.data |> size)")
    factor = (longEdge) / (img.data |> size |> x -> max(x...))
    new_size = map( x -> Int(floor(factor*x)) , img.data |> size)
    resized_img = Images.imresize(img, new_size)
    sample = separate(resized_img).data * 256
    # sub mean
    sample[:,:,1] -= 123.68
    sample[:,:,2] -= 116.779
    sample[:,:,3] -= 103.939
    println("resize the content image to $(new_size)")
    return reshape(sample, (size(sample)[1], size(sample)[2], 3, 1))
end


function preprocess_style_image(img, shape)
    resized_img = Images.imresize(img, (shape[2], shape[1]))
    sample = separate(resized_img).data * 256
    sample[:,:,1] -= 123.68
    sample[:,:,2] -= 116.779
    sample[:,:,3] -= 103.939
    return reshape(sample, (size(sample)[1], size(sample)[2], 3, 1))
end

function postprocess_image(img)
    img = reshape(img, (size(img)[1], size(img)[2], 3))
    img[:,:,1] += 123.68
    img[:,:,2] += 116.779
    img[:,:,3] += 103.939
    img = clamp(img, 0, 255)
    return map(UInt8,(img |> floor))
end

function save_image(img::Array{Float32,4}, filename::AbstractString)
    println("save output to $filename")
    println("dimensions are $(img|>size)")
    out = postprocess_image(img)
    save(filename, colorim(out))
end


#model
type SGExecutor
    executor
    data
    data_grad
end

function style_gram_executor(input_shape, ctx)
    # symbol
    data = mx.Variable("conv")
    rs_data = mx.Reshape(data=data, shape=(Int(prod(input_shape[1:2])),Int(input_shape[3]) ))
    weight = mx.Variable("weight")
    rs_weight = mx.Reshape(data=weight, shape=(Int(prod(input_shape[1:2])),Int(input_shape[3]) ))
    fc = mx.FullyConnected(data=rs_data, weight=rs_weight, no_bias=true, num_hidden=input_shape[3])
    # executor
    conv = mx.zeros(input_shape, ctx)
    grada = mx.zeros(input_shape, ctx)
    args = Dict(:conv => conv, :weight => conv)
    grad = Dict(:conv => grada)
    reqs = Dict(:conv => mx.GRAD_WRITE, :weight => mx.GRAD_NOP )
    executor = mx.bind(fc, ctx, args, args_grad=grad, grad_req=reqs)
    return SGExecutor(executor, conv, grad[:conv])
end

include("model_vgg19.jl")

function train(img::Image, style_image; stop_eps = 0.005, 
				content_weight = 10.0, style_weight = 1.0, max_num_epochs = 100, 
				max_long_edge = 600, LR = 0.1, gpu = 0, save_epochs = 50)

	if gpu == -1
		dev = mx.cpu()
	else
		dev = mx.gpu(0)
	end
	max_long_edge = max(size(img)...)
	content_np = preprocess_content_image(img, max_long_edge)
	style_np = preprocess_style_image(style_image, content_np|> size)
	shape = size(content_np)[1:3]

	model_executor = get_model(shape, dev)
	gram_executor = [style_gram_executor(arr |> size, dev) for arr in model_executor.style]

	# get style representation
	style_array = [mx.zeros(gram.executor.outputs[1] |> size, dev) for gram in gram_executor]
	model_executor.data[:] = style_np
	mx.forward(model_executor.executor)

	for i in 1:length(model_executor.style)
		copy!(gram_executor[i].data,model_executor.style[i])
		mx.forward( gram_executor[i].executor )
		copy!(style_array[i],gram_executor[i].executor.outputs[1])
	end

	# get content representation
	content_array = mx.zeros(model_executor.content |> size, dev)
	content_grad  = mx.zeros(model_executor.content |> size, dev)
	model_executor.data[:] = content_np
	mx.forward(model_executor.executor)
	copy!(content_array,model_executor.content)

	 # train
	img = mx.zeros(content_np |> size, dev)
	img[:] = mx.rand(-0.1, 0.1, img |> size)

	lr = mx.LearningRate.Factor(10, .9, LR)

	optimizer = mx.SGD(
		lr = LR,
		momentum = 0.9,
		weight_decay = 0.005,
		lr_scheduler = lr,
		grad_clip = 10)
	optim_state = mx.create_state(optimizer,0, img)
	optimizer.state = mx.OptimizationState(10)

	old_img = img |> copy
	new_img = old_img

	#p = ProgressMeter.Progress(max_num_epochs, 1)
	for epoch in 1:max_num_epochs
		copy!(model_executor.data,img  )
		mx.forward(model_executor.executor)

		# style gradient
		for i in 1:length(model_executor.style)
			copy!(gram_executor[i].data,model_executor.style[i])
			mx.forward(gram_executor[i].executor)
			mx.backward(gram_executor[i].executor,[gram_executor[i].executor.outputs[1] - style_array[i]])
			mx.div_from!(gram_executor[i].data_grad, (size(gram_executor[i].data)[3] ^2) * (prod(size(gram_executor[i].data)[1:2])))
			mx.mul_to!(gram_executor[i].data_grad, style_weight)
		end

		# content gradient
		mec = model_executor.content |> copy
		@mx.nd_as_jl ro=content_array rw=content_grad begin content_grad[:] = (mec - content_array) * content_weight end

		# image gradient
		grad_array = append!([gram_executor[i].data_grad::mx.NDArray for i in 1:length(gram_executor)] , [content_grad::mx.NDArray])
		mx.backward(model_executor.executor,grad_array)

		mx.update(optimizer,0, img, model_executor.data_grad, optim_state)

		new_img = img |> copy
		eps = vecnorm(old_img - new_img) / vecnorm(new_img)
		old_img = new_img

		if eps < stop_eps
			println("eps < $(stop_eps), training finished")
			break
		end

		if (epoch+1) % save_epochs == 0
			save_image(new_img, "output/tmp_$(string(epoch+1)).jpg")
		end
		#next!(p; showvalues=[(:relative_change, eps)])
	end

	#save_image(new_img, args["output"])
	colorim(postprocess_image(new_img))

end
