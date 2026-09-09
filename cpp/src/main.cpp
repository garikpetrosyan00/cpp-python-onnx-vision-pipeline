#include <atomic>
#include <csignal>
#include <exception>
#include <iostream>

#include "vision/config.hpp"
#include "vision/pipeline.hpp"

namespace {

std::atomic_bool g_stop_requested{false};

extern "C" void handle_interrupt(int) { g_stop_requested.store(true); }

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const vision::CliOptions options = vision::parse_cli(argc, argv);
    if (options.action == vision::CliAction::help) {
      std::cout << vision::help_text();
      return 0;
    }
    if (options.action == vision::CliAction::version) {
      std::cout << "vision_cpp 0.1.0\n";
      return 0;
    }
    std::signal(SIGINT, handle_interrupt);
    const vision::PipelineResult result = vision::run_pipeline(*options.config, g_stop_requested);
    if (result.interrupted) {
      std::cerr << "Interrupted; media resources released.\n";
      return 130;
    }
    return 0;
  } catch (const vision::ConfigError& error) {
    std::cerr << "vision_cpp: error: " << error.what() << "\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "vision_cpp: error: " << error.what() << "\n";
    return 1;
  }
}
