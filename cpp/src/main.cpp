#include <iostream>
#include <string_view>

namespace {

constexpr std::string_view kVersion = "0.1.0";

void print_help() {
    std::cout << "vision_cpp " << kVersion << '\n'
              << "C++/Python ONNX vision pipeline (Phase 0 bootstrap)\n\n"
              << "Usage:\n"
              << "  vision_cpp --help\n"
              << "  vision_cpp --version\n\n"
              << "Media processing and ONNX inference begin in later phases.\n";
}

}

int main(int argument_count, char* arguments[]) {
    if (argument_count == 1) {
        print_help();
        return 0;
    }

    const std::string_view option{arguments[1]};
    if (option == "--help" || option == "-h") {
        print_help();
        return 0;
    }
    if (option == "--version") {
        std::cout << "vision_cpp " << kVersion << '\n';
        return 0;
    }

    std::cerr << "Unknown Phase 0 option: " << option << "\n\n";
    print_help();
    return 2;
}
