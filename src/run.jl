using Images

function preprocess_img(img, shape)
    resized_img = Images.imresize(img, (shape[2], shape[1]))
    sample = separate(resized_img).data * 256
    sample = sample[:,:,1:3]
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

function texturize(img::String, model::String, task::String; output_shape = (500,500), m = 5, out_file = "proc.jpg")
	image = load(img)
	input_size = size(image)
	output_shape = reverse(input_size)
	s1, s2 = output_shape
	s1 = div(s1, 32) * 32
	s2 = div(s2, 32) * 32
	s = generator_symbol(m, task)
    path = joinpath(Pkg.dir("Texturize"), "models")
	args = mx.load("$path/$(model)_args.nd", mx.NDArray)
	auxs = mx.load("$path/$(model)_auxs.nd", mx.NDArray)
	if task == "texture"
		for i = 1:m
			args[Symbol("z_$i")] = mx.rand(-128, 128, (div(s1, 16) * (2^i), div(s2, 16) * (2^i), 3, 1))
		end
	else
		for i = 1:m
			args[Symbol("znoise_$(i-1)")] = mx.rand(-10, 10, (div(s1,16)*(2^(i-1)), div(s2, 16) * (2^(i-1)), 3, 1), mx.cpu(0))
			imager = map(Float32, preprocess_img(image, ( div(s1, 16) * (2^(i-1)), div(s2, 16) * (2^(i-1)), 3, 1)))
			args[Symbol("zim_$(i-1)")] = mx.NDArray(imager)
		end
	end
	m = mx.bind(s, mx.cpu(0), args, aux_states =  auxs)
	mx.forward(m, is_train=true)
	output = postprocess_image(Array{Float32}(m.outputs[1]))
	#colorim(output)
    Images.save(out_file, output) 
end
