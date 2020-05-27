//
// Created by myralllka on 5/24/20.
//

#include "matrix.h"

m_matrix traditional_2D_convolution(std::vector<m_matrix> &src, std::vector<m_matrix> &kernel) {
    size_t result_matrix_size = src[0].get_cols() - kernel[0].get_cols() + 1;
    auto k_size = kernel[0].get_cols();
    float entry;
    m_matrix result(result_matrix_size, result_matrix_size);
    for (size_t counter = 0; counter < src.size(); ++counter) {
        for (size_t i = 0; i < result_matrix_size; ++i) {
            for (size_t j = 0; j < result_matrix_size; ++j) {
                entry = 0;
                for (size_t m = 0; m < k_size; ++m) {
                    for (size_t n = 0; n < k_size; ++n) {
                        entry += kernel[counter].get(m, n) * src[counter].get(i + m, j + n);
                    }
                }
                result.put(i, j, result.get(i, j) + entry);
            }
        }
    }
    return result;
}

m_matrix custom_2D_convolution(std::vector<m_matrix> &src, const std::vector<m_matrix> &kernel) {
    auto patched_image = m_matrix::im2col(src, kernel[0].get_cols());
    auto patched_kernel = m_matrix::im2col(kernel, kernel[0].get_cols());
    auto res = multiply_up_to_3x3_kernel(patched_image, patched_kernel);
    return repatch(res, src[0].get_cols() - kernel[0].get_cols() + 1);
}