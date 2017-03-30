module Texturize

    using Images
    using MXNet

	include("model.jl")
	include("run.jl")
	include("train.jl")
    include("perceptual.jl")

	export texturize, perceptual
end
