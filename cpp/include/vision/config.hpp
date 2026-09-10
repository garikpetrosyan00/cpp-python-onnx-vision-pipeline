#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace vision {

enum class SourceKind { image, video, camera };

struct SourceSpec {
  SourceKind kind;
  int camera_index{-1};
  std::filesystem::path path;
};

struct Config {
  SourceSpec source;
  std::optional<std::filesystem::path> output;
  bool no_display{false};
  std::optional<std::size_t> max_frames;
  double confidence{0.25};
  double iou{0.45};
  std::optional<std::filesystem::path> model;
  std::optional<std::filesystem::path> labels;
  std::vector<std::string> label_names;
  std::optional<std::filesystem::path> detections_json;
  bool benchmark{false};
  std::size_t warmup{5};
  std::optional<std::filesystem::path> benchmark_output;
};

enum class CliAction { run, help, version };

struct CliOptions {
  CliAction action;
  std::optional<Config> config;
};

class ConfigError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

SourceSpec parse_source(const std::string& value);
double parse_unit_interval(const std::string& value, const std::string& option);
std::size_t parse_positive_integer(const std::string& value, const std::string& option);
std::size_t parse_nonnegative_integer(const std::string& value, const std::string& option);
CliOptions parse_cli(int argc, char* argv[]);
std::string help_text();

}  // namespace vision
