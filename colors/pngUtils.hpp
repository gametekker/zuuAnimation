#include <vector>
#include <cstdint>
unsigned char* convertTo8Bit(double* imagecpu, int width, int height);

void writePNG(const char* filename, const std::vector<uint8_t>& image_data, int width, int height);