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

void my_traditional_2D_convolution(matrix &src, matrix &kernel, matrix &result)
{
    size_t a = src.size() - kernel.size() + 1;
    matrix res(a,std::vector<int>(a));
    for (int i = 0; i < a; ++i) {
        for (int j = 0; j < a; ++j) {
            int entry = 0;
            for (int m = 0; m < kernel.size(); ++m) {
                for (int n = 0; n < kernel.size(); ++n) {
                    entry += kernel[m][n]*src[i+m][j+n];
                }
            }
            res[i][j] = entry;
        }
    }
//    print(res);
    result = std::move(res);
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
    my_traditional_2D_convolution(elements, kernel, result);
    print(result);
}
