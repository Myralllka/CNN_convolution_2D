//
// Created by myralllka on 5/23/20.
//

#include <immintrin.h>
#include "../include/matrix.h"

void print_matrix(const matrix &src) {
    for (auto &v : src) {
        for (const float &x : v) std::cout << x << ' ';
        std::cout << std::endl;
    }
}

matrix read_sqr_matrix_from_file(const std::string &filename) {
    matrix buffer;
    size_t x, y, lines = 0;
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
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
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

matrix patch_matrix(const matrix &src, const size_t kernel_size) {
    //    returns NON-Square matrix!
    // patch kernel for optimized convolution
    auto y = std::pow((src.size() - kernel_size + 1), 2);
    auto x = std::pow(kernel_size, 2);
    auto s_size = src.size() - kernel_size + 1;
    size_t counter_x = 0, counter_y = 0;
    matrix result(x, std::vector<float>(y));
    for (size_t i = 0; i < s_size; ++i) {
        for (size_t j = 0; j < s_size; ++j) {
            for (size_t m = 0; m < kernel_size; ++m) {
                for (size_t n = 0; n < kernel_size; ++n) {
                    result[counter_x++][counter_y] = src[i + m][j + n];
                }
            }
            ++counter_y;
            counter_x = 0;
        }
    }
    return result;
}


matrix repatch_matrix(const matrix &src, const size_t res_size) {
    matrix result(res_size, std::vector<float>(res_size));
    size_t i = 0, j = 0;
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

matrix im2col(const std::vector<matrix> &src, const size_t kernel_size) {
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

[[maybe_unused]] matrix multiply(const matrix &first, const matrix &second) {
    size_t number_of_flops = 0;
    matrix result(first.size(), std::vector<float>(second[0].size()));
    auto first_m = first.size(), second_m = second.size(), second_n = second[0].size();
    for (size_t i = 0; i < first_m; ++i) {
        for (size_t k = 0; k < second_m; ++k) {
            for (size_t j = 0; j < second_n; ++j) {
                result[i][j] += first[i][k] * second[k][j];
                ++number_of_flops;
            }
        }
    }
    std::cout << number_of_flops << std::endl;
    return result;
}

matrix row_matrix_on_matrix_multiply_for_3x3_kernel(const matrix &first, const matrix &second) {
    matrix result(first.size(), std::vector<float>(second[0].size()));
    size_t counter = 0;
    size_t number_of_flops = 0;
    m_vector left((first[0].size() / 8 + 1) * 8);
    for (size_t i = 0; i < first[0].size(); ++i) {
        left[i] = first[0][i];
    }
    asm volatile ("# avx code begin");
    // STORE kernel in 4 YMM registers
    __m256 l1 = _mm256_load_ps(left.data);
    __m256 l2 = _mm256_load_ps(left.data + 8);
    __m256 l3 = _mm256_load_ps(left.data + 16);
    __m256 l4 = _mm256_load_ps(left.data + 24);
    asm volatile ("# avx code end");

    for (auto &vec:transpose(second)) {
        m_vector right((vec.size() / 8 + 1) * 8);
        for (size_t i = 0; i < vec.size(); ++i) {
            right[i] = vec[i];
        }
//        asm volatile ("# avx code begin");
        // store one vector of image in other four YMM registers
        __m256 r1 = _mm256_load_ps(right.data);
        __m256 r2 = _mm256_load_ps(right.data + 8);
        __m256 r3 = _mm256_load_ps(right.data + 16);
        __m256 r4 = _mm256_load_ps(right.data + 24);
        // store to right vector result of pairwise multiplications. 4 operations instead of 32
        r1 = _mm256_mul_ps(l1, r1);
        r2 = _mm256_mul_ps(l2, r2);
        r3 = _mm256_mul_ps(l3, r3);
        r4 = _mm256_mul_ps(l4, r4);
        number_of_flops += 4;
        // Now I need to some them all up. 3 operations
        r1 = _mm256_add_ps(r1, r2);
        r2 = _mm256_add_ps(r3, r4);
        r1 = _mm256_add_ps(r1, r2);
//        r1 = _mm256_hadd_ps(r1, r1);
        asm volatile ("# avx code end");
        result[0][counter++] = r1[0] + r1[1] + r1[2] + r1[3] + r1[4] + r1[5] + r1[6] + r1[7];
    }
//    std::cout << number_of_flops << std::endl;
    return result;
}
//
//std::cout << r1[0] << " " << r1[1] << " " << r1[2] << " " << r1[3] << " " << r1[4] << " " << r1[5] << " " << r1[6] << " " << r1[7] << std::endl;
//std::cout << r2[0] << " " << r2[1] << " " << r2[2] << " " << r2[3] << " " << r2[4] << " " << r2[5] << " " << r2[6] << " " << r2[7] << std::endl;
//std::cout << r3[0] << " " << r3[1] << " " << r3[2] << " " << r3[3] << " " << r3[4] << " " << r3[5] << " " << r3[6] << " " << r3[7] << std::endl;
//std::cout << r4[0] << " " << r4[1] << " " << r4[2] << " " << r4[3] << " " << r4[4] << " " << r4[5] << " " << r4[6] << " " << r4[7] << std::endl;