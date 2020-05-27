//
// Created by myralllka on 5/23/20.
//

#include <immintrin.h>
#include "matrix.h"

typedef std::vector<std::vector<float>> matrix;

m_matrix read_sqr_matrix_from_file(const std::string &filename) {
    matrix buffer = std::vector<std::vector<float>>{};
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
    m_matrix result(buffer.size(), buffer.size());
    for (size_t i = 0; i < buffer.size(); ++i) {
        for (size_t j = 0; j < buffer.size(); ++j) {
            result.put(i, j, buffer[i][j]);
        }
    }
    return result;
}

m_matrix repatch(const m_matrix &src, const size_t &res_size) {
    auto t = src.real_transpose();
    m_matrix result(res_size, res_size);
    size_t n = 0, m = 0, counter = 0;
    auto v = src.get_cols(), u = src.get_rows();
    for (int i = 0; i < u; ++i) {
        for (int j = 0; j < v; ++j) {
            ++counter;
            result.put(n, m, src.get(j, i));
            ++n;
            if (n == res_size) {
                ++m;
                n = 0;
            }
        }
    }
    return result;
}

m_matrix multiply_up_to_3x3_kernel(const m_matrix &first, const m_matrix &second) {
    // matrix on vector multiplication.
    size_t res_counter = 0;
    m_matrix result(first.get_cols(), second.get_cols());
    asm volatile ("# avx code begin");
    // STORE kernel in 4 YMM registers
    __m256 l1 = _mm256_load_ps(second.get_data(0));
    __m256 l2 = _mm256_load_ps(second.get_data(8));
    __m256 l3 = _mm256_load_ps(second.get_data(16));
    __m256 l4 = _mm256_load_ps(second.get_data(24));
    asm volatile ("# avx code end");

    for (size_t counter = 0; counter < first.get_cols(); ++counter) {

        asm volatile ("# avx code begin");
        // store one vector of image in other four YMM registers
        __m256 r1 = _mm256_load_ps(first.get_data(counter * 32));
        __m256 r2 = _mm256_load_ps(first.get_data(counter * 32 + 8));
        __m256 r3 = _mm256_load_ps(first.get_data(counter * 32 + 16));
        __m256 r4 = _mm256_load_ps(first.get_data(counter * 32 + 24));
        // store to right vector result of pairwise multiplications. 4 operations instead of 32
        r1 = _mm256_mul_ps(l1, r1);
        r2 = _mm256_mul_ps(l2, r2);
        r3 = _mm256_mul_ps(l3, r3);
        r4 = _mm256_mul_ps(l4, r4);

        // Now I need to some them all up. 3 operations
        r1 = _mm256_add_ps(r1, r2);
        r2 = _mm256_add_ps(r3, r4);
        r1 = _mm256_add_ps(r1, r2);
        asm volatile ("# avx code end");
        result.put(res_counter++, 0, r1[0] + r1[1] + r1[2] + r1[3] + r1[4] + r1[5] + r1[6] + r1[7]);
    }
    return result;
}

//std::cout << r1[0] << " " << r1[1] << " " << r1[2] << " " << r1[3] << " " << r1[4] << " " << r1[5] << " " << r1[6] << " " << r1[7] << std::endl;
//std::cout << r2[0] << " " << r2[1] << " " << r2[2] << " " << r2[3] << " " << r2[4] << " " << r2[5] << " " << r2[6] << " " << r2[7] << std::endl;
//std::cout << r3[0] << " " << r3[1] << " " << r3[2] << " " << r3[3] << " " << r3[4] << " " << r3[5] << " " << r3[6] << " " << r3[7] << std::endl;
//std::cout << r4[0] << " " << r4[1] << " " << r4[2] << " " << r4[3] << " " << r4[4] << " " << r4[5] << " " << r4[6] << " " << r4[7] << std::endl;