using Base.Test
using Texturize
using Images

a = perceptual("$(Pkg.dir("Texturize"))/input/3.jpg", "s4"; out = "thing.jpg")
b = load("thing.jpg")
@test typeof(a[1]) == RGB{N0f8}
@test typeof(b[1]) == RGB{N0f8}
