cmake_minimum_required(VERSION 3.16)
project(CNN_convolution_2D)

set(CMAKE_CXX_STANDARD 20)
# PVS Studio
include(PVS-Studio.cmake)
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#
#if (CMAKE_COMPILER_IS_GNUCC)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
#endif ()

#set(CMAKE_BUILD_TYPE Debug)
file(GLOB_RECURSE SRC "includes/*.h" "src/*.cpp")
set(DCMAKE_CXX_FLAGS=-mavx)

add_executable(CNN_convolution_2D main.cpp ${SRC})


#pvs_studio_add_target(TARGET ${PROJECT_NAME}.analyze ALL
#        ANALYZE CNN_convolution_2D
#        OUTPUT FORMAT errorfile
#        MODE GA:1,2,3 OP:1,2,3 GA:1,2,3 OP:1,2,3 64:1,2,3 CS:1,2,3 MISRA:1,2,3
#        LOG target_${PROJECT_NAME}.err)
#