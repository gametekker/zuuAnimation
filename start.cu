#include "start.hpp"
#include "pngUtils.hpp"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <vector>
#include <array>
#include <thrust/tuple.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <complex>
#include <string>

__device__ cuDoubleComplex cuCcos(cuDoubleComplex z) {
    double x = cuCreal(z);
    double y = cuCimag(z);

    double cos_x = cos(x);
    double sin_x = sin(x);
    double cosh_y = cosh(y);
    double sinh_y = sinh(y);

    return make_cuDoubleComplex(cos_x * cosh_y, -sin_x * sinh_y);
}

__device__ cuDoubleComplex cuCpow(cuDoubleComplex z, cuDoubleComplex w) {
    // For simplicity, assuming w is purely imaginary (w = iy)
    double y = cuCimag(w);

    // Complex logarithm of z
    double r = cuCabs(z);
    double theta = atan2(cuCimag(z), cuCreal(z));
    cuDoubleComplex log_z = make_cuDoubleComplex(log(r), theta);

    // w * log(z)
    cuDoubleComplex wy_logz = make_cuDoubleComplex(-y * theta, y * log(r));

    // e^(wy_logz)
    double exp_part = exp(cuCreal(wy_logz));
    return make_cuDoubleComplex(exp_part * cos(cuCimag(wy_logz)), exp_part * sin(cuCimag(wy_logz)));
}

__device__ cuDoubleComplex f(const cuDoubleComplex& z, double c1, double c2) {
    // Create a complex number equivalent to std::complex<double>(0.0, 1.0)
    cuDoubleComplex i = make_cuDoubleComplex(0.0, c1);

    // Use cuCmul for complex multiplication and cuCcos for cosine
    cuDoubleComplex z_times_i = cuCmul(z, i);
    cuDoubleComplex cos_term = cuCcos(z_times_i);

    // Use cuCpow for complex exponentiation
    cuDoubleComplex exp_term = cuCpow(z, make_cuDoubleComplex(0.0, c2));

    // Use cuCmul for complex multiplication of the results
    return cuCmul(cos_term, exp_term);
}

__global__ void create_complex_grid(int x,
                      int y,
                      double x0, 
                      double xn, 
                      double y0, 
                      double yn, 
                      //"return" field
                      cuDoubleComplex* grid){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    double inc = (xn-x0)/x;

    if (i < x && j < y){
        grid[i + j*x]=make_cuDoubleComplex(x0+i*inc, y0+j*inc);
    }

}

__global__ void create_output (int x,int y, cuDoubleComplex* grid, cuDoubleComplex* device_ptr, int index, double c1, double c2){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < x && j < y){
        device_ptr[index+i + j*x]=f(grid[i + j*x], c1, c2);
    }
}


void find_bin_index(double width, double height, int divX, int divY, double x, double y, int& cx, int& cy) {
    // Calculate the size of each bin
    double bin_width = width / (divX + 1);
    double bin_height = height / (divY + 1);

    // Calculate the bin index for x and y
    int binX = static_cast<int>(std::floor(x / bin_width));
    int binY = static_cast<int>(std::floor(y / bin_height));

    // Ensure the bin indices are within the range
    binX = std::max(0, std::min(binX, divX));
    binY = std::max(0, std::min(binY, divY));

    cx = binX;
    cy = binY;
}

// Function to convert an array of 3 double RGB values to an 8-bit char representation
char convertToCharColor(const double rgbArray[3]) {
    // Ensure the input values are in the range [0.0, 1.0]
    double r = rgbArray[0];
    double g = rgbArray[1];
    double b = rgbArray[2];

    r = std::max(0.0, std::min(1.0, r));
    g = std::max(0.0, std::min(1.0, g));
    b = std::max(0.0, std::min(1.0, b));

    // Convert double values to 8-bit char values (0-255 range)
    uint8_t red = static_cast<uint8_t>(r * 255);
    uint8_t green = static_cast<uint8_t>(g * 255);
    uint8_t blue = static_cast<uint8_t>(b * 255);

    // Combine the color channels into a single char or use separate char values as needed
    // Here, we are packing them into a single char, which is a simple representation.
    return static_cast<char>((red << 16) | (green << 8) | blue);
}
__global__ void ComplexToRGBKernel(const cuDoubleComplex* input, unsigned char* output, int size, int run_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const cuDoubleComplex& c = input[run_idx+idx];
        double magnitude = cuCabs(c);
        unsigned char intensity = static_cast<unsigned char>(magnitude * 255.0 / 10.0); // Example scaling

        // Compute the output index considering the run_idx offset
        int outputIdx = (run_idx + idx) * 3;
        output[outputIdx] = intensity;     // Red channel
        output[outputIdx + 1] = intensity; // Green channel
        output[outputIdx + 2] = intensity; // Blue channel
    }
}

__global__ void ComplexToRGBKerneli(const cuDoubleComplex* input, int* output, int x, int y, int run_idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        const cuDoubleComplex& c = input[run_idx + i + j*x];
        double magnitude = cuCabs(c);
        int intensity = (int)(magnitude * 255.0 / 10.0); // Example scaling

        // Compute the output index considering the run_idx offset
        int outputIdx = (run_idx + i + j*x) * 3;
        output[outputIdx] = intensity;     // Red channel
        output[outputIdx + 1] = intensity; // Green channel
        output[outputIdx + 2] = intensity; // Blue channel
    }
}

void render_frames(double y0, double yn, double x0, double xn, int resolution, double c1, double c2, int chunks){
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device); // Get the current device
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    if (prop.deviceOverlap) {
        std::cout << "Device supports concurrent kernel execution and memory transfers." << std::endl;
    } else {
        std::cout << "Device does not support concurrent kernel execution and memory transfers." << std::endl;
    }

    std::cout << "starting program" << std::endl;

    // Array bounds
    int x = resolution * (xn - x0);
    int y = resolution * (yn - y0);
    dim3 blockDim(x / 32, y / 32);
    dim3 gridDim(32, 32);

    // Allocate memory for grid on device
    cuDoubleComplex* grid;
    cudaMalloc(&grid, x * y * sizeof(cuDoubleComplex));
    create_complex_grid<<<blockDim, gridDim>>>(x, y, y0, yn, x0, xn, grid);
    cudaDeviceSynchronize();
    std::cout << "created complex grid" << std::endl;

    // Allocate memory for colored_image on device
    cuDoubleComplex* function_out_dev;
    cudaMalloc(&function_out_dev, x*y*2*sizeof(cuDoubleComplex));

    int* image_data_dev;
    cudaMalloc(&image_data_dev, x*y*3*2*sizeof(int));
    int* host_image_data = new int[x*y*3*2];

    // Create two streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Create a CUDA event
    //outputs are [c1,c2,c3,c4, ...]
 
    for (int chunk = 0; chunk < chunks+2; ++chunk) {
        
        int run_idx;
        int load_idx;
        if (chunk%2 == 0){
            run_idx=0;
            load_idx=x*y;
        } else {
            run_idx=x*y;
            load_idx=0;
        }

        if (chunk < chunks-2){
            create_output<<<blockDim, gridDim, 0, stream1>>>(x, y, grid, function_out_dev, run_idx, c1, c2);
            ComplexToRGBKerneli<<<blockDim, gridDim, 0, stream1>>>(function_out_dev, image_data_dev, x, y, run_idx);
        }
        if (chunk >= 1 && chunk < chunks-1){
            cudaMemcpyAsync(host_image_data + load_idx*3, 
                image_data_dev + load_idx*3, 
                sizeof(int) * x*y*3, 
                cudaMemcpyDeviceToHost, 
                stream2);
        }
        if (chunk >= 2){
            std::cout<<host_image_data[run_idx*3]<<std::endl;
            std::cout<<host_image_data[run_idx*3 + x*y-1]<<std::endl;
            std::string name = "output"+std::to_string(chunk-2)+".png";
            writePNG(name.c_str(), host_image_data, x, y, x*y);
        }

        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
    }

    // Clean up
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    // Clean up
    cudaFree(grid);
    cudaFree(function_out_dev);
    cudaFree(image_data_dev);
    delete[] host_image_data;
}
