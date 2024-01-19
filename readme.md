# ZuuGPU Project

## Introduction
ZuuGPU is a project that efficiently generates a series of .png files of the graph of a complex function using CUDA. It leverages GPU computation to create high-performance graphical outputs.

# Compilation
`cmake -S . -B build`
`cmake --build build`

# Example Usage
note: this is a work in progress

to run, use command `./build/ZuuGPU 1.0 3.0`

we pass in the coefficients of the complex function (`1.0` `3.0`) as arguments

# Current Status
Very early stages. Capable of generating n frames, but no animation yet.
Efficiently generates a series of `.png` files of the graph of a complex function.
This project currently does the following concurrently using CUDA streams for each frame:
- generate graph of complex function on the GPU
- transfer graph to CPU
- create `.png` file of that graph

# Next Steps
Have the function coefficients change every frame to give output animated look.

