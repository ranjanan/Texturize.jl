using MXNet

type ConvExecutor
    executor :: mx.Executor
    data :: mx.NDArray
    data_grad :: mx.NDArray
    style :: Array{mx.NDArray,1}
    content :: mx.NDArray
end

function get_model(input_size, ctx)
    # declare symbol
    data = mx.Variable("data")
    conv1_1 = mx.Convolution(name="conv1_1", data=data , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu1_1 = mx.Activation(name="relu1_1", data=conv1_1 , act_type="relu")
    conv1_2 = mx.Convolution(name="conv1_2", data=relu1_1 , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu1_2 = mx.Activation(name="relu1_2", data=conv1_2 , act_type="relu")
    pool1 = mx.Pooling(name="pool1", data=relu1_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type="avg")
    conv2_1 = mx.Convolution(name="conv2_1", data=pool1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu2_1 = mx.Activation(name="relu2_1", data=conv2_1 , act_type="relu")
    conv2_2 = mx.Convolution(name="conv2_2", data=relu2_1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu2_2 = mx.Activation(name="relu2_2", data=conv2_2 , act_type="relu")
    pool2 = mx.Pooling(name="pool2", data=relu2_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type="avg")
    conv3_1 = mx.Convolution(name="conv3_1", data=pool2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu3_1 = mx.Activation(name="relu3_1", data=conv3_1 , act_type="relu")
    conv3_2 = mx.Convolution(name="conv3_2", data=relu3_1 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu3_2 = mx.Activation(name="relu3_2", data=conv3_2 , act_type="relu")
    conv3_3 = mx.Convolution(name="conv3_3", data=relu3_2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu3_3 = mx.Activation(name="relu3_3", data=conv3_3 , act_type="relu")
    conv3_4 = mx.Convolution(name="conv3_4", data=relu3_3 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu3_4 = mx.Activation(name="relu3_4", data=conv3_4 , act_type="relu")
    pool3 = mx.Pooling(name="pool3", data=relu3_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type="avg")
    conv4_1 = mx.Convolution(name="conv4_1", data=pool3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu4_1 = mx.Activation(name="relu4_1", data=conv4_1 , act_type="relu")
    conv4_2 = mx.Convolution(name="conv4_2", data=relu4_1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu4_2 = mx.Activation(name="relu4_2", data=conv4_2 , act_type="relu")
    conv4_3 = mx.Convolution(name="conv4_3", data=relu4_2 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu4_3 = mx.Activation(name="relu4_3", data=conv4_3 , act_type="relu")
    conv4_4 = mx.Convolution(name="conv4_4", data=relu4_3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu4_4 = mx.Activation(name="relu4_4", data=conv4_4 , act_type="relu")
    pool4 = mx.Pooling(name="pool4", data=relu4_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type="avg")
    conv5_1 = mx.Convolution(name="conv5_1", data=pool4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu5_1 = mx.Activation(name="relu5_1", data=conv5_1 , act_type="relu")

    # style and content layers
    style = mx.Group(relu1_1, relu2_1, relu3_1, relu4_1, relu5_1)
    content = mx.Group(relu4_2)
    out = mx.Group(style, content)

    # make executor
    arg_shapes, output_shapes, aux_shapes = mx.infer_shape(out, data=(input_size[1], input_size[2], 3, 1))
    arg_names = mx.list_arguments(out)
    arg_dict = Dict(zip(arg_names, [mx.zeros(shape, ctx) for shape in arg_shapes]))
    grad_dict = Dict(zip(arg_names, [mx.zeros(shape, ctx) for shape in arg_shapes]))
    # init with pretrained weight
    pretrained = mx.load("./model/vgg19.params",mx.NDArray)
    for name in arg_names
        if string(name) == "data"
            continue
        end
        key = Symbol("arg:" * string(name))
        copy!(arg_dict[name],pretrained[key])
    end
    executor = mx.bind(out, ctx, arg_dict, args_grad=grad_dict, grad_req=mx.GRAD_WRITE)
    return ConvExecutor(executor,
                        arg_dict[:data],
                        grad_dict[:data],
                        executor.outputs[1:end-1],
                        executor.outputs[end])
end
