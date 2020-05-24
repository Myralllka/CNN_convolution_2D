//
// Created by myralllka on 5/24/20.
//

#include "../include/matrix.h"

matrix traditional_2D_convolution(const std::vector<matrix> &src, const std::vector<matrix> &kernel) {
    int result_matrix_size = src[0].size() - kernel[0].size() + 1;
    auto k_size = kernel[0].size();
    float entry;
    matrix result(result_matrix_size, std::vector<float>(result_matrix_size));
    for (int counter = 0; counter < src.size(); ++counter) {
        for (int i = 0; i < result_matrix_size; ++i) {
            for (int j = 0; j < result_matrix_size; ++j) {
                entry = 0;
                for (int m = 0; m < k_size; ++m) {
                    for (int n = 0; n < k_size; ++n) {
                        entry += kernel[counter][m][n] * src[counter][i + m][j + n];
                    }
                }
                result[i][j] += entry;
            }
        }
    }
    return result;
}


[[maybe_unused]] matrix custom_2D_convolution(const std::vector<matrix> &src, const std::vector<matrix> &kernel) {
    auto patched_image = im2col(src, kernel[0].size());
    auto patched_kernel = kernel2col(kernel);
//    return repatch_matrix(multiply(patched_kernel, patched_image), src[0].size() - kernel[0].size() + 1);
    return repatch_matrix(row_matrix_on_matrix_multiply_for_3x3_kernel(patched_kernel, patched_image),
                          src[0].size() - kernel[0].size() + 1);
}
