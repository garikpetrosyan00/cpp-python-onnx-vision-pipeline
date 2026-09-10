#include "vision/config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <string_view>
#include <fstream>

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

std::vector<std::string> load_labels(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) throw ConfigError("Cannot read --labels file " + path.string() + ". Check path and permissions.");
  std::vector<std::string> labels; std::string line;
  while (std::getline(stream,line)) { while(!line.empty() && (line.back()=='\r'||line.back()==' '||line.back()=='\t')) line.pop_back(); if(line.empty()) throw ConfigError("Labels must contain exactly 80 nonempty unique names in COCO order."); labels.push_back(line); }
  std::set<std::string> unique(labels.begin(),labels.end());
  if(labels.size()!=80 || unique.size()!=80) throw ConfigError("Labels " + path.string() + " must contain exactly 80 nonempty unique names in COCO order.");
  return labels;
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

std::size_t parse_nonnegative_integer(const std::string& value, const std::string& option) {
  if (!digits_only(value)) throw ConfigError(option + " must be a non-negative integer.");
  try {
    const auto parsed = std::stoull(value);
    if (parsed > std::numeric_limits<std::size_t>::max()) {
      throw ConfigError(option + " must be a non-negative integer.");
    }
    return static_cast<std::size_t>(parsed);
  } catch (const std::exception&) {
    throw ConfigError(option + " must be a non-negative integer.");
  }
}

CliOptions parse_cli(int argc, char* argv[]) {
  if (argc == 1) {
    return {CliAction::help, std::nullopt};
  }
  std::optional<std::string> source;
  std::optional<std::filesystem::path> output;
  std::optional<std::filesystem::path> model;
  std::optional<std::filesystem::path> labels;
  std::optional<std::filesystem::path> detections_json;
  std::optional<std::filesystem::path> benchmark_output;
  bool no_display = false;
  bool benchmark = false;
  std::size_t warmup = 5;
  std::optional<std::size_t> max_frames;
  double confidence = 0.25;
  double iou = 0.45;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--help" || argument == "-h") return {CliAction::help, std::nullopt};
    if (argument == "--version") return {CliAction::version, std::nullopt};
    if (argument == "--source") source = option_value(index, argc, argv, argument);
    else if (argument == "--model") model = option_value(index, argc, argv, argument);
    else if (argument == "--labels") labels = option_value(index, argc, argv, argument);
    else if (argument == "--detections-json") detections_json = option_value(index, argc, argv, argument);
    else if (argument == "--benchmark-output") benchmark_output = option_value(index, argc, argv, argument);
    else if (argument == "--output") output = option_value(index, argc, argv, argument);
    else if (argument == "--max-frames") max_frames = parse_positive_integer(option_value(index, argc, argv, argument), argument);
    else if (argument == "--confidence") confidence = parse_unit_interval(option_value(index, argc, argv, argument), argument);
    else if (argument == "--iou") iou = parse_unit_interval(option_value(index, argc, argv, argument), argument);
    else if (argument == "--warmup") warmup = parse_nonnegative_integer(option_value(index, argc, argv, argument), argument);
    else if (argument == "--benchmark") benchmark = true;
    else if (argument == "--no-display") no_display = true;
    else {
      throw ConfigError("Unknown option: " + argument + ". Run --help for supported options.");
    }
  }
  if (!source) return {CliAction::help, std::nullopt};
  if (!model && labels) throw ConfigError("--labels requires --model; omit both for media passthrough.");
  if (detections_json && !model) throw ConfigError("--detections-json requires --model.");
  if (detections_json && detections_json->extension() != ".json") throw ConfigError("--detections-json must be a .json path.");
  if (benchmark && !no_display) throw ConfigError("--benchmark requires --no-display so GUI wait time is never measured.");
  if (benchmark_output && !benchmark) throw ConfigError("--benchmark-output requires --benchmark.");
  if (benchmark_output && lowercase(benchmark_output->extension().string()) != ".json") {
    throw ConfigError("--benchmark-output must be a .json path.");
  }
  if (model) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(*model,error) || error) throw ConfigError("--model file is missing or not a regular file: " + model->string());
    if (!labels) labels = std::filesystem::path("models/classes.txt");
  }
  Config config{parse_source(*source), output, no_display, max_frames, confidence, iou, model, labels,
                {}, detections_json, benchmark, warmup, benchmark_output};
  if (config.detections_json && config.source.kind != SourceKind::image) throw ConfigError("--detections-json currently requires an image source for one canonical document.");
  if (config.labels) config.label_names=load_labels(*config.labels);
  validate_output(config);
  return {CliAction::run, config};
}

std::string help_text() {
  return "vision_cpp 0.1.0\n"
         "C++ media pipeline (CPU YOLOX-Nano detection and benchmarking)\n\n"
         "Usage:\n  vision_cpp --source SOURCE [options]\n\n"
         "Options:\n"
         "  --source SOURCE       Camera index, image, or video file\n"
         "  --model PATH          Audited YOLOX-Nano ONNX model (CPU only)\n"
         "  --labels PATH         80 COCO labels (defaults to models/classes.txt)\n"
         "  --detections-json P   Write canonical detections JSON (model image only)\n"
         "  --benchmark           Measure frame stages; requires --no-display\n"
         "  --warmup N            Unmeasured benchmark frames (default: 5)\n"
         "  --benchmark-output P  Benchmark JSON path (also writes sibling CSV)\n"
         "  --output PATH         Save unchanged image/video frames\n"
         "  --no-display          Run without GUI windows\n"
         "  --max-frames N        Stop after a positive number of frames\n"
         "  --confidence VALUE    Minimum objectness × class score in [0,1]\n"
         "  --iou VALUE           Class-aware NMS IoU threshold in [0,1]\n"
         "  --help, -h            Show this help\n"
         "  --version             Show version\n\n"
         "Press Q or ESC to exit interactive mode. Use --no-display on headless Linux.\n";
}

}  // namespace vision
