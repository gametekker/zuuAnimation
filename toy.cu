#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <cuda_runtime.h>
#include <iostream>

struct square {
    __host__ __device__
    float operator()(const float& x) const { 
        return x + x; 
    }
};

int main() {
    const int N = 10; // Size of the array

    // Create CUDA streams and event
    cudaStream_t stream1, stream2;
    cudaEvent_t event;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaEventCreate(&event);

    // Device array
    thrust::device_vector<float> a(N, 1.0f); // Initialize with 1.0

    // Thrust operation on array a using stream1
    thrust::transform(thrust::cuda::par.on(stream1), a.begin(), a.end(), a.begin(), square());
    //cudaEventRecord(event, stream1);
    
    // Host array allocated manually
    float* b = new float[N];

    // Wait for stream1 to complete in stream2
    //cudaStreamWaitEvent(stream2, event, 0);

    // Copy array a to array b on stream2 using cudaMemcpyAsync
    cudaMemcpyAsync(b, thrust::raw_pointer_cast(a.data()), N * sizeof(float), cudaMemcpyDeviceToHost, stream2);

    // Synchronize stream2 (to ensure the copy is finished)
    //cudaStreamSynchronize(stream2);

    // Destroy streams and event
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(event);

    // Output the results
    for (int i = 0; i < N; ++i) {
        std::cout << "b[" << i << "] = " << b[i] << std::endl;
    }

    // Free the host memory
    delete[] b;

    return 0;
}
