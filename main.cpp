#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

typedef std::vector<std::vector<int>> matrix;

void flip_matrix(matrix &buffer) {
    for (auto &row:buffer) {
        std::reverse(row.begin(), row.end());
    }
    std::reverse(buffer.begin(), buffer.end());
}

void read_sqr_matrix_from_file(const std::string &filename, matrix &buffer) {
    int x, y, lines = 0;
    std::string line;
    std::ifstream in(filename);
    if (!in) {
        std::cout << "Cannot open file.\n";
        return;
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
}

void print(const matrix &buffer) {
    for (auto &v : buffer) {
        for (int x : v) std::cout << x << ' ';
        std::cout << std::endl;
    }
}

void traditional_2D_convolution(matrix &src, matrix &kernel, matrix &result) {
    auto kernel_center= kernel.size() / 2;
    for (int i = 0; i < src.size(); ++i) {
        result.emplace_back();
        for (int j = 0; j < src.size(); ++j) {
            result[i].emplace_back();
            for (int m = 0; m < kernel.size(); ++m) {
                int mm = kernel.size() - 1 - m;
                for (int n = 0; n < kernel.size(); ++n) {
                    int nn = kernel.size() - 1 - n;
                    int ii = i + (kernel_center - mm);
                    int jj = j + (kernel_center - nn);
                    if (ii >= 0 && ii < src.size() && jj >= 0 && jj < src.size())
                        result[i][j] += src[ii][jj] * kernel[mm][nn];
                }
            }
        }
    }
}

void custom_2D_convolution(matrix &src, matrix &kernel, matrix &result) {

}

//void patch_matrix(matrix &buffer) {
//
//    matrix result;
//}

int main() {
    //////////////////////////////////////////////
    std::string input_filename = "input";
    std::string kernel_filename = "kernel";
    //////////////////////////////////////////////
    matrix elements, kernel, result;
    read_sqr_matrix_from_file(kernel_filename, kernel);
    read_sqr_matrix_from_file(input_filename, elements);
    //////////////////////////////////////////////
    traditional_2D_convolution(elements, kernel, result);
    print(result);
}
