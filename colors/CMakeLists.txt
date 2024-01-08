# CMakeLists.txt

# Specify the minimum required CMake version
cmake_minimum_required(VERSION 3.12)

project(colors)

# List the source files containing helper functions
set(SOURCES
    LuminanceColormap.cpp
    pngUtils.cpp
    # Add more source files as needed
)

# Create a library target for helper functions
add_library(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} png)