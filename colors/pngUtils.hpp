#include <vector>
#include <cstdint>
unsigned char* convertTo8Bit(double* imagecpu, int width, int height);

void writePNG(const char* filename, const unsigned char* image_data, int width, int height);