# CNN_convolution_2D
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
### About Convolution in 2D

# Sources
- [article](http://www.songho.ca/dsp/convolution/convolution.html#convolution_2d)
- [This video](https://www.youtube.com/watch?v=_iZ3Q7VXiGI)
- [one more article](https://medium.com/@_init_/an-illustrated-explanation-of-performing-2d-convolutions-using-matrix-multiplications-1e8de8cd2544)
