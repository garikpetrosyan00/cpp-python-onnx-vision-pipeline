#include "test_support.hpp"
#include "vision/config.hpp"

#include <fstream>
#include <vector>

namespace {

vision::CliOptions parse(std::vector<std::string> arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (auto& argument : arguments) argv.push_back(argument.data());
  return vision::parse_cli(static_cast<int>(argv.size()), argv.data());
}

}  // namespace

void run_config_tests() {
  using test_support::check;
  using test_support::expect_throw;
  using test_support::TemporaryDirectory;

  TemporaryDirectory directory;
  const auto image = directory.path() / "fixture.PNG";
  std::ofstream(image) << "fixture";
  const auto video = directory.path() / "fixture.avi";
  std::ofstream(video) << "fixture";
  const auto model = directory.path() / "model.onnx";
  std::ofstream(model) << "fixture";
  const auto labels = directory.path() / "labels.txt";
  {
    std::ofstream label_stream(labels);
    for (int index = 0; index < 80; ++index) label_stream << "class" << index << '\n';
  }

  const auto camera = vision::parse_source("12");
  check(camera.kind == vision::SourceKind::camera && camera.camera_index == 12, "Camera parsing failed.");
  check(vision::parse_source(image.string()).kind == vision::SourceKind::image, "Image parsing failed.");
  check(vision::parse_source(video.string()).kind == vision::SourceKind::video, "Video parsing failed.");
  expect_throw([] { vision::parse_source("-1"); }, "non-negative");
  expect_throw([] { vision::parse_source("missing.png"); }, "does not exist");
  expect_throw([] { vision::parse_source("https://example.test/a.mp4"); }, "Unsupported source URL");
  expect_throw([] { vision::parse_unit_interval("1.1", "--confidence"); }, "between 0 and 1");
  expect_throw([] { vision::parse_unit_interval("nan", "--iou"); }, "between 0 and 1");
  expect_throw([] { vision::parse_positive_integer("0", "--max-frames"); }, "positive integer");
  expect_throw([] { vision::parse_nonnegative_integer("-1", "--warmup"); }, "non-negative");

  const auto options = parse({"vision_cpp", "--source", image.string(), "--output",
                              (directory.path() / "result.jpg").string(), "--no-display", "--max-frames", "1",
                              "--confidence", "0", "--iou", "1"});
  check(options.action == vision::CliAction::run && options.config->no_display, "CLI parsing failed.");
  check(options.config->max_frames == 1U, "Maximum frame parsing failed.");
  check(parse({"vision_cpp"}).action == vision::CliAction::help, "No-argument help failed.");
  check(parse({"vision_cpp", "--no-display"}).action == vision::CliAction::help, "Missing-source help failed.");
  check(parse({"vision_cpp", "--version"}).action == vision::CliAction::version, "Version parsing failed.");
  expect_throw([&] { parse({"vision_cpp", "--source", image.string(), "--output", "result.txt"}); },
               "Unsupported output extension");
  expect_throw([&] { parse({"vision_cpp", "--source", image.string(), "--output", image.string()}); },
               "Output must differ");
  expect_throw([&] { parse({"vision_cpp", "--source", image.string(), "--detections-json", "result.json"}); },
               "requires --model");
  expect_throw([&] { parse({"vision_cpp", "--source", image.string(), "--model", model.string(), "--labels", labels.string(), "--detections-json", "result.txt"}); },
               "must be a .json");
  expect_throw([&] { parse({"vision_cpp", "--source", video.string(), "--model", model.string(), "--labels", labels.string(), "--detections-json", "result.json"}); },
               "requires an image source");
  expect_throw([&] { parse({"vision_cpp", "--source", video.string(), "--benchmark"}); },
               "requires --no-display");
  expect_throw([&] { parse({"vision_cpp", "--source", video.string(), "--no-display", "--benchmark-output", "result.json"}); },
               "requires --benchmark");
  expect_throw([&] { parse({"vision_cpp", "--source", video.string(), "--no-display", "--benchmark", "--benchmark-output", "result.txt"}); },
               "must be a .json");
}
