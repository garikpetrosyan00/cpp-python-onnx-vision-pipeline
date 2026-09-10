#include <exception>
#include <iostream>
#include <string>

void run_config_tests();
void run_input_source_tests();
void run_pipeline_tests();
void run_detector_tests();
void run_detections_json_tests();
int run_real_model_smoke(const std::string& model, const std::string& labels);

int main(int argc, char* argv[]) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--real-smoke") return run_real_model_smoke(argv[2], argv[3]);
    run_config_tests();
    run_input_source_tests();
    run_pipeline_tests();
    run_detector_tests();
    run_detections_json_tests();
    std::cout << "All vision_cpp tests passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return 1;
  }
}
