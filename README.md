# CNN convolution 2D
In CNN architectures, most of the influence time is consumed by Convolution Layers. Besides huge number of multiplications, 2D convolution layer inference is not directly mapped to matrix product.
# Task
Review several methods to transform 2D convolution to matrix product and decrease number of float
multiplications. Implement classic 2D convolution and one of the discovered methods in C++, measure
performance difference. Do not use 3rd party libraries.
# Expected output
- Report with short review of discovered methods, motivation of selecting one of them for
implementations, achieved results
- Source code for traditional 2D convolution implementation
- Source code for optimized 2D convolution implementation
Use 2D convolution (3x3 kernel size) for 100x100x3 input.

# Report
### Convolution in 2D
#### Metric of performance
To compare optimization of algorithm, in this approach it is needed to count the number of float multiplications, FLOP.
#### Traditional (naive) solution
The very simple and direct method is to iterate through every layer of whole matrix and multiply each entry by kernel entry and add it to the result entry. So how to make it faster and with less multiplications? As far as our data in matrix have space locality (the data inside the matrix/vector stores linearly and it is faster to access next cell than randomly chosen one). That\`s why, as it is written in the task, it is better to reduce Convolution to matrix multiplication. 
#### Using Matrix Multiplication or GEMM (generalized matrix multiplication)
I have found in [this](https://cs231n.github.io/convolutional-networks/#conv) article on how to reduce convolution to matrix multiplication. There described that it is necessary to implement such operation as `patch` for every layer, or (in general) operation called `im2col` to reduce 3 channels image to one matrix. And also it is needs to make some operations with the kernel: flatten each kernel and make a matrix that have these vectors as it\`s Row space. It can make the algorithm faster, but not decrease the number of FLOP\`s
The very first idea is to use SIMD approach AVX or SSEx, NVIDIA CUDA.
#### CUDA GEMM optimization
As far as I know from experience, CUDA is very good for big matrices, but herg se I have 100\*100\*3 matrix, so it can be slower to load data into GPU\`s memory and then load back, but it is the best FLOP optimization. 
#### AVX_8 Intel optimization (using YMM registers)
It have to be faster, because YMM registers can store and make operations on 8 float numbers per operation, so it definitely use less operations for multiplications.
#### Winograd's Algorithm
One more very fast approach for fast Convolution is this algorithm, but I have not found any more detail info than [here](https://blog.usejournal.com/understanding-winograd-fast-convolution-a75458744ff). 
# Final custom Implementation
As far as the task is implement 2D CNN as matrix multiplications for 100x100x3 input image and 3x3 kernel, and reduce the number of multiplications, I decided to make it using SIMD approach, YMM x86 registers that can store up to 8 float numbers, using intrinsics for cpp. 
I have made multiplication function only for vector (patched 1 source kernel) on the patched matrix after im2col custom function. I used here a custom matrix class that inside is one dimensional `float*` array, allocated with alignment (I don\`t know and can not find how to make it with `new`).  
# Project structure & usage
### Structure
<b>Image</b> folder contains 3 files, 3 different channels for convolution. <b>Kernel</b> folder contains 3 files, 3 different channels for kernels. If there need to be only one, there have to be three copies of one file.</br>
### Usage
```shell script
chmod +x ./start.sh
./start.sh [options]
``` 
possible options for script:
```shell script
Options:
    -k    --kernel        Directory where kernel located. ./kernel/ by default.
    -m    --image         Directory where image located (one chanel per file). ./image/ by default
    -c    --compile       Compile before executing
    -co   --compileopt    Compile with optimization
    -h    --help          Show help message
```
### Result
Counter of FLOP shows that I used 6 times less multiply operations. 
But, as far as there were such functions as `im2col`, `transpose`, `patch` and `repatch`, that access memory without space locality and are just additional useless operations, I refuse that operations and leave only `im2col`.
So now on 100x100x3 input image and 3x3x3 kernel this implementation times are **53.57sec** for traditional and **34.7sec** for custom one. As perf shows, that `im2col` now takes the most time.
#### What I have changed from yesterday
- Custom matrix class
- Reduce number of allocations because analyze the project better using `perf` and `hotspot`, reduce number of irrelevant allocations and useless functions (for example, instead of transpose I have modified im2col, that now it produced already transposed matrix that is better for multiplication using SIMD.)
- Improve memory allocations, so now there now need to reallocate align memory - after im2col the matrix is totally ready for be loaded into YMM registers that is very fast
- Added time measurement
- (Hope) fixed possibility to compile on another pc
# Important REMARK
In this implementation I used special intrinsics for `Intel x86` processors, used in `row_matrix_on_matrix_multiply_for_3x3_kernel` function (src/matrix.cpp file 133 row). 
To compile it on `intel core i7-7700Hq` there is obligatory __CMake__ command `set(DCMAKE_CXX_FLAGS=-mavx)` that is in __CMakeLists.txt__ row 7. 
CLion warnings that I\`m using non-portable x86_64 intrinsic functions, but no way. 
# Sources
- [video](https://www.youtube.com/watch?v=_iZ3Q7VXiGI)
- [CNN general article](https://cs231n.github.io/convolutional-networks/#conv)
- [CNN general article](http://www.songho.ca/dsp/convolution/convolution.html#convolution_2d)
- [CNN general article](https://medium.com/@_init_/an-illustrated-explanation-of-performing-2d-convolutions-using-matrix-multiplications-1e8de8cd2544)
- [CNN general article](https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/)
- [How to update CNN](https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/)
- [SSE article](https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX)
