#include "include/matrix.h"

int main(int argc, char *argv[]) {
    //////////////////////////////////////////////
    std::string image_path = "image/";
    std::string kernel_path = "kernel/";

    if (argc == 2) {
        image_path = argv[1];
    } else if (argc == 3) {
        image_path = argv[1];
        kernel_path = argv[2];
    } else if (argc > 3) {
        std::cerr << "Too many arguments. Usage:\n"
                     "\t program_name [image_directory] [kernel_directory]" << std::endl;
        exit(1);
    }

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
    print_matrix(traditional_2D_convolution(image, kernel));
    print_matrix(custom_2D_convolution(image, kernel));

}
