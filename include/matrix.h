//
// Created by myralllka on 5/23/20.
//

#ifndef CNN_CONVOLUTION_2D_MATRIX_H
#define CNN_CONVOLUTION_2D_MATRIX_H

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <filesystem>
#include <xmmintrin.h>
//#include <algorithm>


typedef std::vector<std::vector<float>> matrix;

matrix read_sqr_matrix_from_file(const std::string &filename);

void print_matrix(const matrix &src);

matrix traditional_2D_convolution(const std::vector<matrix> &src, const std::vector<matrix> &kernel);

matrix custom_2D_convolution(const std::vector<matrix> &src, const std::vector<matrix> &kernel);

matrix im2col(const std::vector<matrix> &src, const size_t kernel_size);

matrix kernel2col(const std::vector<matrix> &src);

[[maybe_unused]] matrix multiply(const matrix &first, const matrix &second);

matrix repatch_matrix(const matrix &src, const size_t res_size);

matrix row_matrix_on_matrix_multiply_for_3x3_kernel(const matrix &first, const matrix &second);

class m_vector {
    size_t size;
public:
    float *data;

    m_vector(size_t n) : size(n) {
        data = (float *) aligned_alloc(32, size * sizeof(float));
        for (size_t i = 0; i < size; ++i) {
            data[i] = 0;
        }
    }

    ~m_vector() {
        free(data);
    }

    float &operator[](size_t i) {
        return data[i];
    }

    size_t get_size() {
        return size;
    }
};


#endif //CNN_CONVOLUTION_2D_MATRIX_H
