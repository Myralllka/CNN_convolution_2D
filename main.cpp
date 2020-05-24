#include "include/matrix.h"

int main() {
    //////////////////////////////////////////////
    std::string image_path = "image/";
    std::string kernel_path = "kernel/";
    //////////////////////////////////////////////
    std::vector<matrix> kernel;
    std::vector<matrix> image;
    for (const auto &file :std::filesystem::directory_iterator(image_path)) {
        image.emplace_back(std::move(read_sqr_matrix_from_file(file.path())));
    }
    for (const auto &file :std::filesystem::directory_iterator(kernel_path)) {
        kernel.emplace_back(std::move(read_sqr_matrix_from_file(file.path())));
    }
//    for (auto &ch:image) print_matrix(ch);
    print_matrix(custom_2D_convolution(image, kernel));
}
