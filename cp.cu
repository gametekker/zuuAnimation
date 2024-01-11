#include <iostream>
#include <cuda_runtime.h>

// Simple CUDA Kernel for demonstration
__global__ void addKernel(int *d_a, int *d_b, int *d_c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_c[i] = d_a[i] + d_b[i];
    }
}

int main() {
    const int size = 1024;
    int *h_a, *h_b, *h_c; // Host arrays
    int *d_a, *d_b, *d_c; // Device arrays

    // Allocate host memory
    h_a = new int[size];
    h_b = new int[size];
    h_c = new int[size];

    // Initialize host arrays
    for (int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device memory
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));

    // Create a CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Copy data from host to device asynchronously
    cudaMemcpyAsync(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Launch kernel in the stream
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, size);

    // Copy result back to host asynchronously
    cudaMemcpyAsync(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost, stream);

    // Synchronize the stream (wait for all operations in the stream to complete)
    cudaStreamSynchronize(stream);

    // Host-side processing after synchronization
    std::cout << "Results after kernel execution:" << std::endl;
    for (int i = 0; i < 5; i++) { // Print first 5 results
        std::cout << "h_c[" << i << "] = " << h_c[i] << std::endl;
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaStreamDestroy(stream);

    return 0;
}
