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
#include "speed_tester.h"
#include "m_matrix.h"

m_matrix read_sqr_matrix_from_file(const std::string &filename);

m_matrix traditional_2D_convolution(std::vector<m_matrix> &src, std::vector<m_matrix> &kernel);

m_matrix custom_2D_convolution(std::vector<m_matrix> &src, const std::vector<m_matrix> &kernel);

m_matrix multiply_up_to_3x3_kernel(const m_matrix &first, const m_matrix &second);

m_matrix repatch(const m_matrix &src, const size_t &res_size);

#endif //CNN_CONVOLUTION_2D_MATRIX_H
