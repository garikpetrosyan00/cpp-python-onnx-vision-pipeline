#include "test_support.hpp"
#include "vision/detections_json.hpp"

#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>

void run_detections_json_tests() {
  using test_support::check;
  using test_support::expect_throw;
  using test_support::TemporaryDirectory;
  TemporaryDirectory directory;
  const auto destination = directory.path() / "nested" / "detections.json";
  const std::vector<vision::Detection> detections{
      {3, "control\n\t\x01\"\\", 0.5F, 1.25F, 2.5F, 3.75F, 4.0F, 7}};
  vision::write_detections_json(destination, detections, 416, 416, 0.01, 0.45);
  std::ifstream stream(destination);
  const std::string json((std::istreambuf_iterator<char>(stream)), {});
  check(json.find("\"schema_version\":\"vision-pipeline-detections/v1\"") != std::string::npos,
        "Canonical schema version is missing.");
  check(json.find("\"implementation\":\"cpp\"") != std::string::npos,
        "C++ implementation identity is missing.");
  check(json.find("\"class_id\":3,\"label\":\"control\\u000a\\u0009\\u0001\\\"\\\\\"") !=
            std::string::npos,
        "JSON control-character escaping failed.");
  check(json.find("\"confidence\":0.5,\"x1\":1.25,\"y1\":2.5,\"x2\":3.75,\"y2\":4") !=
            std::string::npos,
        "Canonical detection field order changed.");

  std::ofstream(destination, std::ios::trunc) << "existing";
  auto invalid = detections;
  invalid[0].x1 = std::numeric_limits<float>::quiet_NaN();
  expect_throw([&] { vision::write_detections_json(destination, invalid, 416, 416, 0.01, 0.45); },
               "finite numeric values");
  std::ifstream preserved(destination);
  check(std::string((std::istreambuf_iterator<char>(preserved)), {}) == "existing",
        "Failed export replaced an existing destination.");
  std::size_t files = 0;
  for (const auto& entry : std::filesystem::directory_iterator(destination.parent_path())) {
    ++files;
    check(entry.path() == destination, "Failed export left a temporary file.");
  }
  check(files == 1, "Expected only the preserved destination after failure.");

  const auto blocked = directory.path() / "blocked.json";
  std::filesystem::create_directory(blocked);
  expect_throw([&] { vision::write_detections_json(blocked, detections, 416, 416, 0.01, 0.45); },
               "Cannot publish");
  check(std::filesystem::is_directory(blocked), "Publish failure replaced the existing destination.");
  for (const auto& entry : std::filesystem::directory_iterator(directory.path()))
    check(entry.path().extension() != ".tmp", "Publish failure left a temporary file.");
}
