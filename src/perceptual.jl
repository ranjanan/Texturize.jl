const ws = 8192
function block(data, num_filter; name = "thing")
    data2 = conv(data, num_filter, 1, name=name)
    data2 = mx.Convolution(mx.SymbolicNode, data=data2, num_filter=num_filter, kernel=(3,3), pad=(1,1), name="$(name)_conv1", workspace = ws)
    data2 = mx.BatchNorm(mx.SymbolicNode, data=data2, momentum=0.9, name="$(name)_bn1")
    mx.Activation(mx.SymbolicNode, data=data+data2, act_type="relu")
end

function conv(data, num_filter, stride; name = "thing")
    data = mx.Convolution(mx.SymbolicNode, data=data, num_filter=num_filter, kernel=(3,3), pad=(1,1), stride=(stride, stride), name="$(name)_conv", workspace = ws)
    data = mx.BatchNorm(mx.SymbolicNode, data=data, momentum=0.9, name="$(name)_conv")
    data = mx.Activation(mx.SymbolicNode, data=data, act_type="relu")
    return data
end

function generator_symbol()
    data = mx.Variable("data")
    data = mx.Convolution(mx.SymbolicNode, data=data, num_filter=32, kernel=(9,9), pad=(4,4), name="conv0", workspace = ws)
    data = mx.BatchNorm(mx.SymbolicNode, data=data, name="bn0")
    data = mx.Activation(mx.SymbolicNode, data=data, act_type="relu")
    data = conv(data, 64, 2, name="downsample0")
    data = conv(data, 128, 2, name="downsample1")
    data = block(data, 128, name="block0")
    data = block(data, 128, name="block1")
    data = block(data, 128, name="block2")
    data = block(data, 128, name="block3")
    data = block(data, 128, name="block4")
    data = mx.Deconvolution(mx.SymbolicNode, data=data, kernel=(4,4), pad=(0,0), stride=(2,2), num_filter=64, name="deconv0")
    data = mx.BatchNorm(mx.SymbolicNode, data=data, momentum=0.9, name="dcbn0")
    data = mx.Activation(mx.SymbolicNode, data=data, act_type="relu")
    data = mx.Deconvolution(mx.SymbolicNode, data=data, kernel=(4,4), pad=(0,0), stride=(2,2), num_filter=32, name="deconv1")
    data = mx.BatchNorm(mx.SymbolicNode, data=data, momentum=0.9, name="dcbn1")
    data = mx.Activation(mx.SymbolicNode, data=data, act_type="relu")
    data = mx.Convolution(mx.SymbolicNode, data=data, num_filter=3, kernel=(9,9), pad=(1,1), name="lastconv", workspace = ws)
    return data 
end


function descriptor_symbol(style_layers=["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"], content_layer="relu4_2")
    data = mx.Variable("data")
    conv1_1 = mx.Convolution(mx.SymbolicNode, name="conv1_1", data=data , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=ws)
    relu1_1 = mx.Activation(mx.SymbolicNode, name="relu1_1", data=conv1_1 , act_type="relu")
    conv1_2 = mx.Convolution(mx.SymbolicNode, name="conv1_2", data=relu1_1 , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=ws)
    relu1_2 = mx.Activation(mx.SymbolicNode, name="relu1_2", data=conv1_2 , act_type="relu")
    pool1 = mx.Pooling(mx.SymbolicNode, name="pool1", data=relu1_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type="avg")
    conv2_1 = mx.Convolution(mx.SymbolicNode, name="conv2_1", data=pool1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=ws)
    relu2_1 = mx.Activation(mx.SymbolicNode, name="relu2_1", data=conv2_1 , act_type="relu")
    conv2_2 = mx.Convolution(mx.SymbolicNode, name="conv2_2", data=relu2_1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=ws)
    relu2_2 = mx.Activation(mx.SymbolicNode, name="relu2_2", data=conv2_2 , act_type="relu")
    pool2 = mx.Pooling(mx.SymbolicNode, name="pool2", data=relu2_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type="avg")
    conv3_1 = mx.Convolution(mx.SymbolicNode, name="conv3_1", data=pool2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=ws)
    relu3_1 = mx.Activation(mx.SymbolicNode, name="relu3_1", data=conv3_1 , act_type="relu")
    conv3_2 = mx.Convolution(mx.SymbolicNode, name="conv3_2", data=relu3_1 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=ws)
    relu3_2 = mx.Activation(mx.SymbolicNode, name="relu3_2", data=conv3_2 , act_type="relu")
    conv3_3 = mx.Convolution(mx.SymbolicNode, name="conv3_3", data=relu3_2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=ws)
    relu3_3 = mx.Activation(mx.SymbolicNode, name="relu3_3", data=conv3_3 , act_type="relu")
    conv3_4 = mx.Convolution(mx.SymbolicNode, name="conv3_4", data=relu3_3 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=ws)
    relu3_4 = mx.Activation(mx.SymbolicNode, name="relu3_4", data=conv3_4 , act_type="relu")
    pool3 = mx.Pooling(mx.SymbolicNode, name="pool3", data=relu3_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type="avg")
    conv4_1 = mx.Convolution(mx.SymbolicNode, name="conv4_1", data=pool3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=ws)
    relu4_1 = mx.Activation(mx.SymbolicNode, name="relu4_1", data=conv4_1 , act_type="relu")
    conv4_2 = mx.Convolution(mx.SymbolicNode, name="conv4_2", data=relu4_1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=ws)
    relu4_2 = mx.Activation(mx.SymbolicNode, name="relu4_2", data=conv4_2 , act_type="relu")
    conv4_3 = mx.Convolution(mx.SymbolicNode, name="conv4_3", data=relu4_2 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=ws)
    relu4_3 = mx.Activation(mx.SymbolicNode, name="relu4_3", data=conv4_3 , act_type="relu")
    conv4_4 = mx.Convolution(mx.SymbolicNode, name="conv4_4", data=relu4_3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=ws)
    relu4_4 = mx.Activation(mx.SymbolicNode, name="relu4_4", data=conv4_4 , act_type="relu")
    pool4 = mx.Pooling(mx.SymbolicNode, name="pool4", data=relu4_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type="avg")
    conv5_1 = mx.Convolution(mx.SymbolicNode, name="conv5_1", data=pool4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=ws)
    relu5_1 = mx.Activation(mx.SymbolicNode, name="relu5_1", data=conv5_1 , act_type="relu")
    style_out = mx.Group([map(x -> eval(Symbol(x)), style_layers)])
    return mx.Group([style_out, eval(Symbol(content_layer))])
end

function perceptual(img, model_prefix, gpu = -1, save = true)
	image = load(img)
    output_shape = size(image)
	s2, s1 = output_shape
	s1 = div(s1, 32) * 32
	s2 = div(s2, 32) * 32
    s = generator_symbol()

    path = joinpath(Pkg.dir("Texturize"), "models")
    args = mx.load("$path/$(model_prefix)_args.nd", mx.NDArray)
    auxs = mx.load("$path/$(model_prefix)_auxs.nd", mx.NDArray)

    imager = preprocess_img(image, (s1, s2))

    args[:data] = mx.NDArray(imager)

	ctx = gpu == -1 ? mx.cpu() : mx.gpu()
	m = mx.bind(s, mx.cpu(), args, aux_states =  auxs)
	mx.forward(m, is_train=true)

	output = postprocess_img(Array{Float32}(m.outputs[1]))
	final_output = colorview(RGB{N0f8}, output)

	save("$(img)_output.jpg", final_output)

	final_output
end
