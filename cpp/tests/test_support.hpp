#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace test_support {

inline void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
void expect_throw(Callable callable, const std::string& expected) {
  try {
    callable();
  } catch (const std::exception& error) {
    check(std::string(error.what()).find(expected) != std::string::npos,
          "Expected error containing '" + expected + "', got '" + error.what() + "'.");
    return;
  }
  throw std::runtime_error("Expected an exception containing '" + expected + "'.");
}

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    path_ = std::filesystem::temp_directory_path() / ("vision_cpp_test_" + std::to_string(getpid()) + "_" +
                                                       std::to_string(++counter_));
    std::filesystem::create_directories(path_);
  }
  ~TemporaryDirectory() { std::filesystem::remove_all(path_); }
  const std::filesystem::path& path() const { return path_; }

 private:
  inline static int counter_{0};
  std::filesystem::path path_;
};

}  // namespace test_support
