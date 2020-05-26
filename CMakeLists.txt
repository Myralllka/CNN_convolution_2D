cmake_minimum_required(VERSION 3.16)
project(CNN_convolution_2D)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE SRC "includes/*.h" "src/*.cpp")
set(DCMAKE_CXX_FLAGS=-mavx)
set(DCMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(CNN_convolution_2D main.cpp ${SRC})
