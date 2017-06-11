# Texturize.jl

This package implements an algorithm called neural styles. 

Currently, you can generate an image using any of available pre-trained models. 

## Usage

```julia
perceptual("path/to/input/", "model"; gpu = -1, write = true, out = "")
```

Kwargs: 
* gpu: If you set this to `true`, the model is loaded onto the GPU. Improves performance. 
* write: If this is true, it writes to file. 
* out: By default, `path/$(name)_output.jpg` is the name of the file written. With this,
you can change the output filename and path. 

## Example

If this is the input: 

![alt text](https://github.com/ranjanan/Texturize.jl/blob/master/input/example.jpg)

running `perceptual("img.jpg", "s4")` will give you: 

![alt text](https://github.com/ranjanan/Texturize.jl/blob/master/input/example_output.jpg)

## References

* [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

## Credits

The original Python implementation [here](https://github.com/zhaw/neural_style) by @zhaw

