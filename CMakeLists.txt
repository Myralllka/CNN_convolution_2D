cmake_minimum_required(VERSION 3.16)
project(CNN_convolution_2D)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)
file(GLOB_RECURSE SRC "includes/*.h" "src/*.cpp")

add_executable(CNN_convolution_2D main.cpp ${SRC})