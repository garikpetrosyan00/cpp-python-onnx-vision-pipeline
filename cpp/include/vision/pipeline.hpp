#pragma once

#include <atomic>

#include "vision/config.hpp"

namespace vision {

struct PipelineResult {
  std::size_t frames{0};
  bool interrupted{false};
};

PipelineResult run_pipeline(const Config& config, const std::atomic_bool& stop_requested);

}  // namespace vision
