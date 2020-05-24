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

typedef std::vector<std::vector<int>> matrix;

matrix read_sqr_matrix_from_file(const std::string &filename);

void print_matrix(const matrix &src);

matrix traditional_2D_convolution(const matrix &src, const matrix &kernel);

matrix custom_2D_convolution(const std::vector<matrix> &src, const std::vector<matrix> &kernel);


#endif //CNN_CONVOLUTION_2D_MATRIX_H
