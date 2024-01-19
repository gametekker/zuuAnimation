# Compilation
`cmake -S . -B build`
`cmake --build build`

# Example Usage
note: this is a work in progress
to run, use command `./build/ZuuGPU`

# Current Status
Very early stages. Capable of generating n frames, but no animation yet.
Efficiently generates a series of `.png` files of the graph of a complex function.
This project currently does the following concurrently using CUDA streams for each frame:
- generate graph of complex function on the GPU
- transfer graph to CPU
- create `.png` file of that graph

# Next Steps
Have the function coefficients change every frame to give output animated look.

