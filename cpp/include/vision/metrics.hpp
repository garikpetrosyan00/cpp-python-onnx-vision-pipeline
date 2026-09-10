#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "vision/config.hpp"

namespace vision {

struct FrameTiming {
  std::int64_t capture_ns{};
  std::int64_t preprocess_ns{};
  std::int64_t inference_ns{};
  std::int64_t postprocess_ns{};
  std::int64_t render_ns{};
  std::int64_t total_ns{};
};

struct TimingSummary {
  std::size_t count{};
  std::optional<double> mean_ms;
  std::optional<double> median_ms;
  std::optional<double> p50_ms;
  std::optional<double> p95_ms;
  std::optional<double> p99_ms;
  std::optional<double> min_ms;
  std::optional<double> max_ms;
};

double percentile(const std::vector<std::int64_t>& samples, int percent);
TimingSummary timing_summary(const std::vector<std::int64_t>& samples);

class MetricsCollector {
 public:
  void add(FrameTiming timing);
  const std::vector<FrameTiming>& frames() const noexcept { return frames_; }
  std::optional<double> fps() const;
  TimingSummary summary(const std::string& name) const;

 private:
  std::vector<FrameTiming> frames_;
};

std::filesystem::path default_benchmark_output();
void write_benchmark_results(const std::filesystem::path& destination, const Config& config,
                             const MetricsCollector& collector, std::size_t completed_warmup);

}  // namespace vision
