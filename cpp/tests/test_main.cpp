#include <exception>
#include <iostream>

void run_config_tests();
void run_input_source_tests();
void run_pipeline_tests();

int main() {
  try {
    run_config_tests();
    run_input_source_tests();
    run_pipeline_tests();
    std::cout << "All vision_cpp tests passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return 1;
  }
}
