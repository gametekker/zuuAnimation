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

unsigned char* prepareImageDataForPNG(cuDoubleComplex* h_colored_image, int x, int y, int frames, int chunks, LuminanceColormap cmap, bool old, double vmax, int divX, int divY){
    unsigned char* ca = (unsigned char*) malloc(3*x*y*frames*chunks*sizeof(char));
    for (int image = 0; image < x*y*frames*chunks; image+= x*y){
        for (int i = 0; i < x*y; i++){
            cuDoubleComplex z = h_colored_image[image+i];
            int cx;
            int cy;
            if (old){
                double magnitude;
                double phase;
                // Assuming z is your cuDoubleComplex variable
                double a = cuCreal(z); // Real part of z
                double b = cuCimag(z); // Imaginary part of z
                // Calculate magnitude
                magnitude = sqrt(a * a + b * b);
                // Calculate phase
                phase = atan2(b, a);
                if (phase < 0){
                    phase+=2*M_PI;
                }
                find_bin_index(2*M_PI, vmax, divX, divY, phase, magnitude, cx, cy);
            }
            else{
                cx = cuCreal(z);
                cy = cuCreal(z);
            }
            std::array<double,3> color = cmap.getColor(cx, cy);
            
            // Assuming that color components are in the range [0.0, 1.0]
            // and need to be scaled to [0, 255]
            ca[(image + i) * 3 + 0] = static_cast<unsigned char>(color[0] * 255.0);
            ca[(image + i) * 3 + 1] = static_cast<unsigned char>(color[1] * 255.0);
            ca[(image + i) * 3 + 2] = static_cast<unsigned char>(color[2] * 255.0);
        }
    }
    return ca;
}

void render_single_image(double y0, double yn, double x0, double xn, int resolution, LuminanceColormap cmap, double c1, double c2, int frames, int chunks){
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
    cuDoubleComplex* device_colored_image;
    cudaMalloc(&device_colored_image, x * y * frames * sizeof(cuDoubleComplex));

    // Allocate memory for colored_image on host
    cuDoubleComplex* host_colored_image = new cuDoubleComplex[x * y * frames * chunks];

    // Create two streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    for (int cpu_chunk = 0; cpu_chunk < x * y * frames * chunks; cpu_chunk += x * y * frames) {
        // Process first half
        if (cpu_chunk != 0) {
            std::cout << "precopyx" << std::endl;
            cudaMemcpyAsync(host_colored_image + cpu_chunk + x * y * frames / 2, 
                            device_colored_image + x * y * frames / 2, 
                            sizeof(cuDoubleComplex) * x * y * frames / 2, 
                            cudaMemcpyDeviceToHost, 
                            stream1);
        }

        for (int i = 0; i < x * y * frames / 2; i += x * y) {
            create_output<<<blockDim, gridDim, 0, stream2>>>(x, y, grid, device_colored_image, i, c1, c2);
        }

        // Process second half
        std::cout << "precopy" << std::endl;
        cudaMemcpyAsync(host_colored_image + cpu_chunk, 
                        device_colored_image, 
                        sizeof(cuDoubleComplex) * x * y * frames / 2, 
                        cudaMemcpyDeviceToHost, 
                        stream2);

        std::cout << "postcopy" << std::endl;
        for (int i = x * y * frames / 2; i < x * y * frames; i += x * y) {
            create_output<<<blockDim, gridDim, 0, stream1>>>(x, y, grid, device_colored_image, i, c1, c2);
        }
    }
    std::cout << "precopyx" << std::endl;
            cudaMemcpyAsync(host_colored_image + x * y * frames * chunks - x * y * frames / 2, 
                            device_colored_image + x * y * frames / 2, 
                            sizeof(cuDoubleComplex) * x * y * frames / 2, 
                            cudaMemcpyDeviceToHost, 
                            stream1);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    try {
        std::cout<<"now done"<<std::endl;
        std::cout<<"(-2,-2)" <<cuCreal(host_colored_image[0])<<","<<cuCimag(host_colored_image[0])<<std::endl;
        std::cout<<"(-2,-2)" <<cuCreal(host_colored_image[x*y])<<","<<cuCimag(host_colored_image[x*y])<<std::endl;
        std::cout<<"(2,2)" <<cuCreal(host_colored_image[x*y-1])<<","<<cuCimag(host_colored_image[x*y-1])<<std::endl;
        auto image_data = prepareImageDataForPNG(host_colored_image, x, y, frames, chunks, cmap, true, 40000, cmap.getShape().first, cmap.getShape().second);
        std::cout << "making output image" << std::endl;
        writePNG("output.png", image_data, x, y * frames * chunks);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // Clean up
    cudaFree(grid);
    cudaFree(device_colored_image);
    delete[] host_colored_image;
}
