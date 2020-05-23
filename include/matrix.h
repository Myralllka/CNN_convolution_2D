//
// Created by myralllka on 5/23/20.
//

#ifndef CNN_CONVOLUTION_2D_MATRIX_H
#define CNN_CONVOLUTION_2D_MATRIX_H

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

typedef std::vector<std::vector<int>> matrix;

void read_sqr_matrix_from_file(const std::string &filename, matrix &buffer);

void print_matrix(const matrix &src);

matrix traditional_2D_convolution(const matrix &src, const matrix &kernel);

matrix custom_2D_convolution(const matrix &src, const matrix &kernel);


#endif //CNN_CONVOLUTION_2D_MATRIX_H
