using MXNet

function block(data, num_filter; name = "thing")
    data = mx.Convolution(mx.SymbolicNode, data=data, num_filter=num_filter, kernel=(3,3), pad=(1,1), name="$(name)_conv1")
    data = mx.BatchNorm(mx.SymbolicNode, data=data, momentum=0., name="$(name)_batchnorm1")
    data = mx.LeakyReLU(mx.SymbolicNode, data=data, slope=0.1)
    data = mx.Convolution(mx.SymbolicNode, data=data, num_filter=num_filter, kernel=(3,3), pad=(1,1), name="$(name)_conv2")
    data = mx.BatchNorm(mx.SymbolicNode, data=data, momentum=0., name="$(name)_batchnorm2")
    data = mx.LeakyReLU(mx.SymbolicNode, data=data, slope=0.1)
    data = mx.Convolution(mx.SymbolicNode, data=data, num_filter=num_filter, kernel=(1,1), pad=(0,0), name="$(name)_conv3")
    data = mx.BatchNorm(mx.SymbolicNode, data=data, momentum=0., name="$(name)_batchnorm3")
    data = mx.LeakyReLU(mx.SymbolicNode, data=data, slope=0.1)
    return data
end

function join(data, data_low, num_filter; name = "thing")
    data_low = mx.UpSampling(mx.SymbolicNode, data_low, scale=2, num_filter=num_filter, sample_type="nearest", num_args=1)
    data_low = mx.BatchNorm(mx.SymbolicNode, data=data_low, momentum=0, name="$(name)_batchnorm_low")
    data = mx.BatchNorm(mx.SymbolicNode, data=data, momentum=0, name="$(name)_batchnorm")
    out = mx.Concat(data, data_low)
end

function generator_symbol(m, task)
    Z = mx.SymbolicNode[]
    for i = 1:m
        if task == "texture"
            push!(Z, block(mx.Variable("z_$(i-1)"), 8, name="block$(i-1)"))
        else
            noise = mx.Variable("znoise_$(i-1)")
            r = mx.Variable("zim_$(i-1)")
            push!(Z, block(mx.Concat(noise, r), 8, name="block$(i-1)"))
		end
	end
    for i = 2:m
        Z[i] = block(join(Z[i], Z[i-1], (i-1)*8, name="join$(i-1)"), 8*(i), name="blockjoin$(i-1)")
	end
    out = mx.Convolution(mx.SymbolicNode, data=Z[end], num_filter=3, kernel=(1,1), pad=(0,0), name="blockout")
    return out
end


