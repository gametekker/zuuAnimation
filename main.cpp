#include "start.hpp"
#include "pngUtils.hpp"
#include "pngUtils.hpp"

int main(int argc, char *argv[]){

    // simple setup
    // the complex function itself is hard coded in the kernel

    // we define the coefficients of the complex function by passing them in as args
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        args.push_back(std::string(argv[i]));
    }
    double c0 = std::stod(args[0]);
    double c1 = std::stod(args[1]);
    
    // we specify the range of the function, and the resoltuion

    // range of the function
    double y0 = -2.0;
    double yn = 2.0;
    double x0 = -2.0;
    double xn = 2.0;
    
    int resolution = 512; // must be multiple of 32

    // specify the number of frames
    int num_frames = 10;
    
    render_frames(y0,yn,x0,xn,resolution,c0,c1,num_frames);

}
