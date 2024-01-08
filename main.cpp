#include "start.hpp"
#include "pngUtils.hpp"

int main(int argc, char *argv[]){

    // Create a vector to hold the arguments as std::string
    std::vector<std::string> args;

    // Start from 1 to skip the program name
    for (int i = 1; i < argc; ++i) {
        args.push_back(std::string(argv[i]));
    }

    double y0 = -2.0;
    double yn = 2.0;
    double x0 = -2.0;
    double xn = 2.0;

    int resolution = 1000;

    // Define a simple colormap
    std::vector<std::array<double, 3>> colors = {
        {236.0/255, 244.0/255, 214.0/255},   // Soft Green
        {154.0/255, 208.0/255, 194.0/255},   // Aquamarine
        {45.0/255, 149.0/255, 150.0/255},    // Teal
        {38.0/255, 80.0/255, 115.0/255},     // Deep Sky Blue
        {34.0/255, 9.0/255, 44.0/255},       // Dark Purple
        {135.0/255, 35.0/255, 65.0/255},     // Crimson
        {190.0/255, 49.0/255, 68.0/255},     // Raspberry
        {240.0/255, 89.0/255, 65.0/255},     // Coral
        {7.0/255, 102.0/255, 173.0/255},     // Cobalt Blue
        {41.0/255, 173.0/255, 178.0/255}     // Turquoise
    };
    LuminanceColormap cmap (colors, 10);

    render_single_image(y0,yn,x0,xn,resolution,cmap, std::stod(args[0]), std::stod(args[1]));



}