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
        return 1;
    }

    //////////////////////////////////////////////
    std::vector<m_matrix> kernel;
    std::vector<m_matrix> image;
    for (const auto &file :std::filesystem::directory_iterator(image_path)) {
        image.emplace_back(read_sqr_matrix_from_file(file.path()));
    }
    for (const auto &file :std::filesystem::directory_iterator(kernel_path)) {
        kernel.push_back(read_sqr_matrix_from_file(file.path()));
    }
    auto start_time = get_current_time_fenced();
    for (int i = 0; i < 100000; ++i) {
        auto cus = custom_2D_convolution(image, kernel);
    }
    auto finish_time = get_current_time_fenced();
    std::cout << "Total for custom: " << to_us(finish_time - start_time) << std::endl;
    start_time = get_current_time_fenced();
    for (int i = 0; i < 100
    000; ++i) {
        auto trad = traditional_2D_convolution(image, kernel);
    }
    finish_time = get_current_time_fenced();
    std::cout << "Total for traditional: " << to_us(finish_time - start_time) << std::endl;
    return 0;
}
