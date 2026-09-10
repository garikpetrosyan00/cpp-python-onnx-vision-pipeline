#include "test_support.hpp"
#include "vision/metrics.hpp"

#include <cmath>
#include <fstream>
#include <iterator>

namespace {

vision::Config passthrough_config() {
  vision::Config config{};
  config.source = {vision::SourceKind::video, -1, "fixture.avi"};
  config.no_display = true;
  config.max_frames = 2;
  config.confidence = 0.25;
  config.iou = 0.45;
  config.model = std::nullopt;
  config.labels = std::nullopt;
  config.label_names = {};
  config.detections_json = std::nullopt;
  config.benchmark = true;
  config.warmup = 1;
  config.benchmark_output = std::nullopt;
  return config;
}

}  // namespace

void run_metrics_tests() {
  using test_support::check;
  using test_support::expect_throw;
  using test_support::TemporaryDirectory;

  check(std::abs(vision::percentile({1, 2, 9, 10}, 50) - 5.5) < 1e-12,
        "Percentile median changed.");
  check(std::abs(vision::percentile({1, 2, 9, 10}, 95) - 9.85) < 1e-12,
        "Percentile interpolation changed.");
  expect_throw([] { (void)vision::percentile({}, 50); }, "Invalid percentile");

  vision::MetricsCollector collector;
  collector.add({10, 0, 0, 0, 5, 15});
  collector.add({20, 0, 0, 0, 5, 25});
  check(collector.summary("preprocess").mean_ms == 0.0,
        "Passthrough detector stages must be exactly zero.");
  check(std::abs(*collector.fps() - 50'000'000.0) < 1e-6, "Effective FPS changed.");
  expect_throw([&] { collector.add({1, 2, 3, 4, 5, 1}); }, "Invalid frame timing");

  TemporaryDirectory directory;
  const auto destination = directory.path() / "nested" / "result.json";
  const auto config = passthrough_config();
  vision::write_benchmark_results(destination, config, collector, 1);
  std::ifstream stream(destination);
  const std::string json((std::istreambuf_iterator<char>(stream)), {});
  check(json.find("\"schema_version\":\"1.0\"") != std::string::npos,
        "Benchmark schema is missing.");
  check(json.find("\"implementation\":\"cpp\"") != std::string::npos,
        "C++ benchmark identity is missing.");
  check(json.find("\"preprocess\":{\"count\":2,\"mean_ms\":0") != std::string::npos,
        "Passthrough benchmark stages are not zero.");
  const auto csv = destination.parent_path() / "result.csv";
  check(std::filesystem::is_regular_file(csv), "Benchmark CSV was not published.");

  std::ofstream(destination, std::ios::trunc) << "existing-json";
  std::ofstream(csv, std::ios::trunc) << "existing-csv";
  const auto blocked = directory.path() / "blocked.json";
  std::filesystem::create_directory(blocked);
  expect_throw([&] { vision::write_benchmark_results(blocked, config, collector, 1); },
               "not a regular file");
  check(std::filesystem::is_directory(blocked),
        "A failed benchmark publication replaced an existing destination.");
  std::ifstream preserved(destination);
  check(std::string((std::istreambuf_iterator<char>(preserved)), {}) == "existing-json",
        "A failed publication changed an unrelated existing JSON result.");
  for (const auto& entry : std::filesystem::recursive_directory_iterator(directory.path())) {
    check(entry.path().filename().string().find(".tmp") == std::string::npos,
          "Benchmark publication failure left a temporary file.");
  }
}
