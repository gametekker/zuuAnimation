# ZuuGPU Project

## Introduction
ZuuGPU is a project that efficiently generates a series of .png files of the graph of a complex function using CUDA. It leverages GPU computation to create high-performance graphical outputs.

the function to be plotted is f(z) = z * (c0)i * z ^ (c1)i

where z is a complex number

- the pixel location in the image corresponds to the input value in the complex plane
- the pixel brightness corresponds to the magnitude of f(z)

# Compilation
`cmake -S . -B build`
`cmake --build build`

# Example Usage

to run, use command `./build/ZuuGPU 1.0 3.0`

we pass in the coefficients of the complex function (c0, c1) = (1.0, 3.0) as arguments

# Current Status
Very early stages. Capable of generating n frames, but no animation yet.
Efficiently generates a series of `.png` files of the graph of a complex function.
This project currently does the following concurrently using CUDA streams for each frame:
- generate graph of complex function on the GPU
- transfer graph to CPU
- create `.png` file of that graph

# Next Steps
- Implement coefficient changes every frame to produce an animated output.

![Alt Text](https://github.com/gametekker/zuuAnimation/blob/simplify/out.png)


