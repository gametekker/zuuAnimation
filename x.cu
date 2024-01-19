#include <cuda_runtime.h>
#include <iostream>

__global__ void simpleKernel(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1;  // A simple operation, e.g., increment each element
    }
}

int main() {
    const int size = 1024;
    int *hostData = nullptr, *deviceData = nullptr;

    // Allocate host memory
    hostData = new int[size];

    // Initialize host data
    for (int i = 0; i < size; ++i) {
        hostData[i] = i;
    }

    // Allocate device memory
    cudaMalloc(&deviceData, size * sizeof(int));

    // Create two CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Copy data from host to device asynchronously in stream1
    cudaMemcpyAsync(deviceData, hostData, size * sizeof(int), cudaMemcpyHostToDevice, stream1);

    // Launch kernel in stream2
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    simpleKernel<<<gridSize, blockSize, 0, stream2>>>(deviceData, size);

    // Copy data back from device to host asynchronously in stream1
    cudaMemcpyAsync(hostData, deviceData, size * sizeof(int), cudaMemcpyDeviceToHost, stream1);

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Cleanup
    cudaFree(deviceData);
    delete[] hostData;
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
