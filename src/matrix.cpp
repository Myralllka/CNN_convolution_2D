//
// Created by myralllka on 5/23/20.
//

#include "../include/matrix.h"
#include <immintrin.h>

int TRADITIONAL_FLOP;
int MUL_FLOP;

void print_matrix(const matrix &src) {
    for (auto &v : src) {
        for (int x : v) std::cout << x << ' ';
        std::cout << std::endl;
    }
}

matrix read_sqr_matrix_from_file(const std::string &filename) {
    matrix buffer;
    int x, y, lines = 0;
    std::string line;
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Cannot open file " << filename << std::endl;
        exit(-1);
    }
    while (getline(in, line)) ++lines;
    in.close();
    in.open(filename);
    for (y = 0; y < lines; y++) {
        buffer.emplace_back();
        for (x = 0; x < lines; x++) {
            buffer[y].emplace_back();
            in >> buffer[y][x];
        }
    }
    in.close();
    return buffer;
}

[[maybe_unused]] matrix transpose(const matrix &src) {
//     transpose vector or matrix
    auto n = src.size();
    auto m = src[0].size();
    matrix result(m, std::vector<float>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            result[j][i] = src[i][j];
        }
    }
    return result;
}

[[maybe_unused]] void flip_matrix(matrix &src) {
    for (auto &row:src) {
        std::reverse(row.begin(), row.end());
    }
    std::reverse(src.begin(), src.end());
}

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
                        ++TRADITIONAL_FLOP;
                    }
                }
                result[i][j] += entry;
            }
        }
    }
    std::cout << "traditional: " << TRADITIONAL_FLOP << std::endl;
    return result;
}

matrix patch_matrix(const matrix &src, const int kernel_size) {
    //    returns NON-Square matrix!
    // patch kernel for optimized convolution
    auto y = std::pow((src.size() - kernel_size + 1), 2);
    auto x = std::pow(kernel_size, 2);
    auto s_size = src.size() - kernel_size + 1;
    int counter_x = 0, counter_y = 0;
    matrix result(x, std::vector<float>(y));
    for (int i = 0; i < s_size; ++i) {
        for (int j = 0; j < s_size; ++j) {
            for (int m = 0; m < kernel_size; ++m) {
                for (int n = 0; n < kernel_size; ++n) {
                    result[counter_x++][counter_y] = src[i + m][j + n];
                }
            }
            ++counter_y;
            counter_x = 0;
        }
    }
    return result;
}

matrix multiply(const matrix &first, const matrix &second) {
    matrix result(first.size(), std::vector<float>(second[0].size()));
    auto first_m = first.size(), second_m = second.size(), second_n = second[0].size();
    for (int i = 0; i < first_m; ++i) {
        for (int j = 0; j < second_n; ++j) {
            for (int k = 0; k < second_m; ++k) {
                result[i][j] += first[i][k] * second[k][j];
                ++MUL_FLOP;
            }
        }
    }
    std::cout << "better: " << MUL_FLOP << std::endl;
    return result;
}

matrix repatch_matrix(const matrix &src, const int res_size) {
    matrix result(res_size, std::vector<float>(res_size));
    int i = 0, j = 0;
    for (auto &row: src) {
        for (auto &element:row) {
            result[i][j] = element;
            ++j;
            if (j == res_size) {
                ++i;
                j = 0;
            }
        }
    }
    return result;
}

matrix im2col(const std::vector<matrix> &src, const int kernel_size) {
    matrix result;
    for (auto &ch:src) {
        for (std::vector<float> &entry: patch_matrix(ch, kernel_size)) {
            result.emplace_back(entry);
        }
    }
    return result;
}

matrix kernel2col(const std::vector<matrix> &src) {
    matrix result;
    for (auto &ch:src) {
        for (std::vector<float> &entry: patch_matrix(ch, ch.size())) {
            result.emplace_back(entry);
        }
    }
    return transpose(result);
}



[[maybe_unused]] matrix custom_2D_convolution(const std::vector<matrix> &src, const std::vector<matrix> &kernel) {
    auto patched_image = im2col(src, kernel[0].size());
    auto patched_kernel = kernel2col(kernel);
    return repatch_matrix(multiply(patched_kernel, patched_image), src[0].size() - kernel[0].size() + 1);
}
