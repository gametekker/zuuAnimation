#include "start.hpp"
#include "pngUtils.hpp"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <vector>
#include <thrust/tuple.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

struct max_complex_mag {
    __host__ __device__
    cuDoubleComplex operator()(const cuDoubleComplex& a, const cuDoubleComplex& b) const {
        if (cuCabs(a) > cuCabs(b)) return a;
        else return b;
    }
};

struct Segments {
    int x_bins, y_bins;
    double x_interval_size, y_interval_size;
    int width;

    Segments(double minX, double maxX, double minY, double maxY, int x_bins, int y_bins, int width) 
    : x_bins(x_bins), y_bins(y_bins), width(width) {
        x_interval_size = (maxX - minX) / x_bins;
        y_interval_size = (maxY - minY) / y_bins;
    }

    __device__ thrust::tuple<int,int> operator()(const cuDoubleComplex& element, const size_t i) const {

        int binx_index, biny_index;
        map(atan2(cuCreal(element), cuCimag(element)),cuCabs(element),binx_index,biny_index);

        // Example operation using 2D coordinates
        return thrust::make_tuple (binx_index, biny_index);
    }

    __device__ void map(double x, double y, int &x_index, int &y_index) const {
        x_index = int((x - (-M_PI)) / x_interval_size);  // Adjust for negative start of interval
        y_index = int(y / y_interval_size);

        // Ensure indices are within bounds
        x_index = x_index >= x_bins ? x_bins - 1 : x_index;
        y_index = y_index >= y_bins ? y_bins - 1 : y_index;
    }
};

std::vector<uint8_t> prepareImageDataForPNG(const thrust::host_vector<thrust::tuple<double, double, double>>& colored_image, int width, int height) {

    // Convert to 8-bit unsigned int format
    std::vector<uint8_t> image_data;
    image_data.reserve(width * height * 4); // Reserve space for width x height pixels with 3 components (RGB)
    for (const auto& pixel : colored_image) {
        image_data.push_back(static_cast<uint8_t>(thrust::get<0>(pixel) * 255)); // Red
        image_data.push_back(static_cast<uint8_t>(thrust::get<1>(pixel) * 255)); // Green
        image_data.push_back(static_cast<uint8_t>(thrust::get<2>(pixel) * 255)); // Blue
        image_data.push_back(static_cast<uint8_t>(255)); // Blue
    }

    return image_data;
}

struct ColorMapFunctor {
    double* color_bins;
    int bins_per_color;

    ColorMapFunctor(double* color_bins, int bins_per_color)
        : color_bins(color_bins), bins_per_color(bins_per_color) {}

    __device__ thrust::tuple<double, double, double> operator()(const thrust::tuple<int, int>& index_tuple) {
        int bin_index = (thrust::get<0>(index_tuple) * bins_per_color + thrust::get<1>(index_tuple)) * 3;
        return thrust::make_tuple(
            color_bins[bin_index],     // Red
            color_bins[bin_index + 1], // Green
            color_bins[bin_index + 2] // Blue
        );
    }
};

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

struct OutputFunctor {
    double c1; 
    double c2;
    OutputFunctor(double c1, double c2)
        : c1(c1), c2(c2) {}

    __device__ cuDoubleComplex operator()(const cuDoubleComplex& z) {
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
};

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

void create_output(int index, cuDoubleComplex* grid, int x, int y, LuminanceColormap cmap, OutputFunctor f, 
                   thrust::device_vector<thrust::tuple<double,double,double>>& colored_image, cudaStream_t& stream){
    //NOTE: create outputs (int start, int end, params)
    //perform create output in a series [start ... end] on colored_image
    //transfer to cpu (start ... end) on colored_image
    // [0 ... 1/2](0 ... 1/2)
    //            [1/2 ... 1](1/2 ... 1)
    //                       [0 ... 1/2](0 ... 1/2)
    //                                  [1/2 ... 1](1/2 ... 1)
    // for f in frames:
    //    if f % 2 == 0:
    //        create output (0 ... 1/2)(f)
    //        transfer to cpu ( 1 ... 1/2)(f-1) 
    //    else:
    //        create output (1/2 ... 1)(f)
    //        transfer to cpu ( 0 ... 1/2)(f-1)
    //    syncronize

    cuDoubleComplex* outputs;
    cudaMalloc(&outputs, x * y * sizeof(cuDoubleComplex));
    //make_image<<<blockDim,gridDim>>>(x,y,grid,outputs);
    //cudaDeviceSynchronize();
    
    std::cout<<"created image"<<std::endl;
    thrust::device_ptr<cuDoubleComplex> grid_dp(grid);
    thrust::device_vector<cuDoubleComplex> grid_da(grid_dp,grid_dp + x*y);
    thrust::device_ptr<cuDoubleComplex> outputs_dp(outputs);
    thrust::device_vector<cuDoubleComplex> outputs_da(outputs_dp,outputs_dp + x*y);

    thrust::transform(thrust::cuda::par.on(stream), grid_da.begin(), grid_da.end(), outputs_da.begin(), f);

    //thrust::host_vector<cuDoubleComplex> outputs_ha (outputs_da);
    //std::cout<<cuCreal(outputs_ha[0])<<std::endl;
    //std::cout<<cuCimag(outputs_ha[0])<<std::endl;

    cuDoubleComplex maxv = thrust::reduce(thrust::cuda::par.on(stream), outputs_da.begin(), outputs_da.end(), make_cuDoubleComplex(0, 0), max_complex_mag());
    std::cout<<"compute maximum value"<<std::endl;

    // Create a device vector to store the 3D results
    thrust::device_vector<thrust::tuple<int, int>> skeleton(x*y);
    // Apply the transformation
    thrust::transform(
        thrust::cuda::par.on(stream),
        outputs_da.begin(), outputs_da.end(),
        thrust::counting_iterator<size_t>(0),
        skeleton.begin(),
        //double minX, double maxX, double minY, double maxY, int x_bins, int y_bins, int width
        Segments(-M_PI, M_PI, 0, cuCabs(maxv), cmap.getShape().first, cmap.getShape().second, x)
    );

    //thrust::host_vector<thrust::tuple<int,int>> skeleton_h(skeleton);
    //std::cout<<thrust::get<0>(skeleton_h[10000])<<std::endl;
    //std::cout<<thrust::get<1>(skeleton_h[10000])<<std::endl;

    std::vector<std::vector<std::array<double,3>>> color_bins = cmap.getColormap();
    // Flatten the colormap
    thrust::device_vector<double> d_color_bins;
    for (const auto& bin : color_bins) {
        for (const auto& color : bin) {
            d_color_bins.insert(d_color_bins.end(), color.begin(), color.end());
        }
    }
    // Color the image
    thrust::transform(
        thrust::cuda::par.on(stream),
        skeleton.begin(), skeleton.end(),
        colored_image.begin() + index,
        ColorMapFunctor(thrust::raw_pointer_cast(d_color_bins.data()), color_bins.size())
    );

    //std::cout<<"making output image"<<std::endl;
    //thrust::host_vector<thrust::tuple<double,double,double>> colored_image_h (colored_image);
    //std::cout<<thrust::get<0>(colored_image_h[0])<<std::endl;
    //std::cout<<thrust::get<1>(colored_image_h[0])<<std::endl;
    //std::cout<<thrust::get<2>(colored_image_h[0])<<std::endl;

    cudaFree(outputs);
}

void render_single_image(double y0, double yn, double x0, double xn, int resolution, LuminanceColormap cmap, double c1, double c2){

    std::cout<<"starting program"<<std::endl;

    //array bounds
    int x = resolution*(xn-x0);
    int y = resolution*(yn-y0);
    dim3 blockDim (x/32,y/32);
    dim3 gridDim (32,32);
    std::cout<<x<<std::endl;
    std::cout<<y<<std::endl;
    thrust::device_vector<thrust::tuple<double, double, double>> colored_image(x*y);

    //NOTE: initgrid ( ... )
    cuDoubleComplex* grid;
    cudaMalloc(&grid, x * y * sizeof(cuDoubleComplex));
    create_complex_grid<<<blockDim,gridDim>>>(x,y,y0,yn,x0,xn,grid);
    cudaDeviceSynchronize();
    std::cout<<"created complex grid"<<std::endl;

    //int index, cuDoubleComplex* grid, int x, int y, dim3 blockDim, dim3 gridDim, LuminanceColormap cmap, 
    //thrust::device_vector<thrust::tuple<double,double,double>> colored_image
    // Create two streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    create_output(0, grid, x, y, cmap, OutputFunctor(c1,c2), colored_image, stream1);
    cudaStreamSynchronize(stream1);

    // Retrieve the last error
    cudaError_t err = cudaGetLastError();

    // Check if there was an error
    if (err != cudaSuccess) {
        // Print the error message
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy data from device to host
    std::cout<<"precopy"<<std::endl;
    thrust::host_vector<thrust::tuple<double, double, double>> h_colored_image (x*y); 
    //cudaStreamSynchronize(stream2);
    thrust::copy(thrust::cuda::par.on(stream2), colored_image.begin(), colored_image.end(), h_colored_image.begin());
    std::cout<<"postcopy"<<std::endl;

    try {
        auto image_data = prepareImageDataForPNG(h_colored_image, x, y);
        std::cout<<"making output image"<<std::endl;
        writePNG("output.png", image_data, x, y);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // Clean up streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaFree(grid);

}
