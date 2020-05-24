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
#### Traditional solution
The very simple and direct method is to iterate throw every layer of whole matrix and multiply each entry by kernel entry and add it to the result entry. It is ![formula](https://render.githubusercontent.com/render/math?math=O(N^4)) for ![formula](https://render.githubusercontent.com/render/math?math=N*N) matrix and ![formula](https://render.githubusercontent.com/render/math?math=M*M) kernel where ![formula](https://render.githubusercontent.com/render/math?math=M=\dfrac{N}{2}).
So how to make it and faster and with less multiplications?
#### Using Matrix Multiplication
I have found in [this](https://cs231n.github.io/convolutional-networks/#conv) article how to reduce convolution to matrix multiplication.
It is necessary to implement such operation as `patch` for every layer, or (in general) operation called `im2col`. 
And also it is needs to make some operations with the kernel: flatten each kernel and make a matrix that have these vectors as it`s Row space. 


# Sources
- [video](https://www.youtube.com/watch?v=_iZ3Q7VXiGI)
- [CNN article](https://cs231n.github.io/convolutional-networks/#conv)
- [CNN article](http://www.songho.ca/dsp/convolution/convolution.html#convolution_2d)
- [CNN article](https://medium.com/@_init_/an-illustrated-explanation-of-performing-2d-convolutions-using-matrix-multiplications-1e8de8cd2544)
- [CNN article](https://www.allaboutcircuits.com/technical-articles/two-dimensional-convolution-in-image-processing/)
- [CNN article](https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/)
- [SSE article](https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX)
