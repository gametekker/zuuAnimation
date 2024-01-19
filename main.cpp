#include "start.hpp"
#include "pngUtils.hpp"
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

    int resolution = 512; //must be multiple of 32

    render_frames(y0,yn,x0,xn,resolution, std::stod(args[0]), std::stod(args[1]), 10);



}
