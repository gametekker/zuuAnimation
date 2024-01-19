#include "pngUtils.hpp"
#include <png.h>
#include <stdlib.h>

unsigned char* convertTo8Bit(double* imagecpu, int width, int height) {
    unsigned char* imageData = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));
    for (int i = 0; i < width * height * 4; ++i) {
        imageData[i] = static_cast<unsigned char>(imagecpu[i] * 255);
    }
    return imageData;
}

#include <png.h>
#include <vector>
#include <stdexcept>

void writePNG(const char* filename, const int* rgb_data, int width, int height, int offset) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        throw std::runtime_error("Failed to open file for writing");
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        throw std::runtime_error("Failed to create PNG write structure");
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        fclose(fp);
        throw std::runtime_error("Failed to create PNG info structure");
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        throw std::runtime_error("Failed to set PNG error handling");
    }

    png_init_io(png_ptr, fp);

    // Set the PNG header information for 24-bit RGB (not RGBA)
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    // Allocate temporary memory for converted data
    std::vector<unsigned char> temp_data(width * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                temp_data[(y * width + x) * 3 + c] = static_cast<unsigned char>(rgb_data[(offset + y * width + x) * 3 + c]);
            }
        }
    }

    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; ++y) {
        row_pointers[y] = &temp_data[y * width * 3];
    }

    png_set_rows(png_ptr, info_ptr, row_pointers.data());
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}


