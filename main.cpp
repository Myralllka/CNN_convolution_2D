#include "include/matrix.h"

int main() {
    //////////////////////////////////////////////
    std::string input_filename = "input";
    std::string kernel_filename = "kernel";
    //////////////////////////////////////////////
    matrix elements, kernel, result;
    read_sqr_matrix_from_file(kernel_filename, kernel);
    read_sqr_matrix_from_file(input_filename, elements);
    //////////////////////////////////////////////
    result = traditional_2D_convolution(elements, kernel);
    print_matrix(result);
    result = custom_2D_convolution(elements, kernel);
    print_matrix(result);
}
