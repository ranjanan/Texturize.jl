module Texturize

    using Images
    using MXNet

	include("model.jl")
	include("run.jl")
	include("train.jl")
    include("perceptual.jl")
	import .Perceptual: perceptual

	export texturize, perceptual
end
