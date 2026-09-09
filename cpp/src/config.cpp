#include "vision/config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <string_view>

namespace vision {
namespace {

const std::set<std::string> kImageExtensions{".bmp", ".jpeg", ".jpg", ".pbm", ".pgm", ".png",
                                             ".ppm", ".tif", ".tiff", ".webp"};
const std::set<std::string> kVideoExtensions{".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg",
                                             ".mpg", ".webm", ".wmv"};
const std::set<std::string> kVideoOutputs{".avi", ".mkv", ".mov", ".mp4", ".webm"};

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool digits_only(const std::string& value) {
  return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char c) {
    return std::isdigit(c) != 0;
  });
}

std::string option_value(int& index, int argc, char* argv[], const std::string& option) {
  if (++index >= argc) {
    throw ConfigError(option + " requires a value.");
  }
  return argv[index];
}

void validate_output(const Config& config) {
  if (!config.output) {
    return;
  }
  const auto& output = *config.output;
  const auto extension = lowercase(output.extension().string());
  const auto& extensions = config.source.kind == SourceKind::image ? kImageExtensions : kVideoOutputs;
  if (extensions.count(extension) == 0U) {
    throw ConfigError("Unsupported output extension for selected source: " + output.string());
  }
  std::error_code error;
  const bool output_exists = std::filesystem::exists(output, error);
  if (error) {
    throw ConfigError("Cannot access output path " + output.string() + ": " + error.message());
  }
  if (output_exists && !std::filesystem::is_regular_file(output, error)) {
    throw ConfigError("Output is not a regular file: " + output.string());
  }
  if (error) {
    throw ConfigError("Cannot access output path " + output.string() + ": " + error.message());
  }
  if (output_exists && config.source.kind != SourceKind::camera) {
    if (std::filesystem::equivalent(output, config.source.path, error)) {
      throw ConfigError("Output must differ from the source file; choose another path.");
    }
    if (error) {
      throw ConfigError("Cannot compare output and source: " + error.message());
    }
  }
}

}  // namespace

SourceSpec parse_source(const std::string& value) {
  if (value.empty()) {
    throw ConfigError("--source must be a non-negative camera index or an image/video file path.");
  }
  if (digits_only(value)) {
    try {
      const auto parsed = std::stoll(value);
      if (parsed > std::numeric_limits<int>::max()) {
        throw ConfigError("Camera index is too large; use an index between 0 and 2147483647.");
      }
      return {SourceKind::camera, static_cast<int>(parsed), {}};
    } catch (const std::out_of_range&) {
      throw ConfigError("Camera index is too large; use an index between 0 and 2147483647.");
    }
  }
  if (value.size() > 1 && (value.front() == '-' || value.front() == '+') &&
      digits_only(value.substr(1))) {
    throw ConfigError("Camera index must be a non-negative integer, such as --source 0.");
  }
  if (value.find("://") != std::string::npos) {
    throw ConfigError("Unsupported source URL; use a local image/video file or a camera index.");
  }
  const std::filesystem::path path(value);
  std::error_code error;
  const bool exists = std::filesystem::exists(path, error);
  if (error) {
    throw ConfigError("Cannot access source path " + path.string() + ": " + error.message());
  }
  if (!exists) {
    throw ConfigError("Source file does not exist: " + path.string() + ". Check the --source path.");
  }
  const bool regular = std::filesystem::is_regular_file(path, error);
  if (error) {
    throw ConfigError("Cannot access source path " + path.string() + ": " + error.message());
  }
  if (!regular) {
    throw ConfigError("Source is not a regular file: " + path.string());
  }
  const auto extension = lowercase(path.extension().string());
  if (kImageExtensions.count(extension) != 0U) {
    return {SourceKind::image, -1, path};
  }
  if (kVideoExtensions.count(extension) != 0U) {
    return {SourceKind::video, -1, path};
  }
  throw ConfigError("Unsupported source extension " + (extension.empty() ? "(none)" : extension) +
                    ": " + path.string() + ". Use an image such as PNG/JPEG or video such as AVI/MP4.");
}

double parse_unit_interval(const std::string& value, const std::string& option) {
  std::size_t consumed = 0;
  double parsed = 0.0;
  try {
    parsed = std::stod(value, &consumed);
  } catch (const std::exception&) {
    throw ConfigError(option + " must be a finite number between 0 and 1.");
  }
  if (consumed != value.size() || !std::isfinite(parsed) || parsed < 0.0 || parsed > 1.0) {
    throw ConfigError(option + " must be a finite number between 0 and 1.");
  }
  return parsed;
}

std::size_t parse_positive_integer(const std::string& value, const std::string& option) {
  if (!digits_only(value)) {
    throw ConfigError(option + " must be a positive integer.");
  }
  try {
    const auto parsed = std::stoull(value);
    if (parsed == 0 || parsed > std::numeric_limits<std::size_t>::max()) {
      throw ConfigError(option + " must be a positive integer.");
    }
    return static_cast<std::size_t>(parsed);
  } catch (const std::exception&) {
    throw ConfigError(option + " must be a positive integer.");
  }
}

CliOptions parse_cli(int argc, char* argv[]) {
  if (argc == 1) {
    return {CliAction::help, std::nullopt};
  }
  std::optional<std::string> source;
  std::optional<std::filesystem::path> output;
  bool no_display = false;
  std::optional<std::size_t> max_frames;
  double confidence = 0.25;
  double iou = 0.45;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--help" || argument == "-h") return {CliAction::help, std::nullopt};
    if (argument == "--version") return {CliAction::version, std::nullopt};
    if (argument == "--source") source = option_value(index, argc, argv, argument);
    else if (argument == "--output") output = option_value(index, argc, argv, argument);
    else if (argument == "--max-frames") max_frames = parse_positive_integer(option_value(index, argc, argv, argument), argument);
    else if (argument == "--confidence") confidence = parse_unit_interval(option_value(index, argc, argv, argument), argument);
    else if (argument == "--iou") iou = parse_unit_interval(option_value(index, argc, argv, argument), argument);
    else if (argument == "--no-display") no_display = true;
    else if (argument == "--model" || argument == "--labels" || argument == "--benchmark" ||
             argument == "--warmup" || argument == "--benchmark-output") {
      throw ConfigError(argument + " is unavailable: ONNX inference and benchmarking begin in Phase 5/7.");
    } else {
      throw ConfigError("Unknown option: " + argument + ". Run --help for supported Phase 4 options.");
    }
  }
  if (!source) return {CliAction::help, std::nullopt};
  Config config{parse_source(*source), output, no_display, max_frames, confidence, iou};
  validate_output(config);
  return {CliAction::run, config};
}

std::string help_text() {
  return "vision_cpp 0.1.0\n"
         "C++ media pipeline (Phase 4 passthrough; no AI inference)\n\n"
         "Usage:\n  vision_cpp --source SOURCE [options]\n\n"
         "Options:\n"
         "  --source SOURCE       Camera index, image, or video file\n"
         "  --output PATH         Save unchanged image/video frames\n"
         "  --no-display          Run without GUI windows\n"
         "  --max-frames N        Stop after a positive number of frames\n"
         "  --confidence VALUE    Validate finite [0,1]; unused until Phase 5\n"
         "  --iou VALUE           Validate finite [0,1]; unused until Phase 5\n"
         "  --help, -h            Show this help\n"
         "  --version             Show version\n\n"
         "Press Q or ESC to exit interactive mode. Use --no-display on headless Linux.\n";
}

}  // namespace vision
