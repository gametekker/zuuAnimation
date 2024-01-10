cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# set the project name and version
project(ZuuGPU LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 61)

add_library(start STATIC start.cu)

include_directories("colors")
add_subdirectory("colors")

target_link_libraries(start PRIVATE colors)

add_executable(ZuuGPU main.cpp)

find_package(CUDA REQUIRED)
target_link_libraries(ZuuGPU PRIVATE start ${CUDA_LIBRARIES})

# Add a custom target for cuda-memcheck
add_custom_target(memcheck
    COMMAND cuda-memcheck $<TARGET_FILE:ZuuGPU>
    DEPENDS ZuuGPU
)