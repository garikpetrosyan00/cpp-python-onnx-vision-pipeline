#include "vision/pipeline.hpp"

#include <stdexcept>

#include "vision/input_source.hpp"
#include "vision/renderer.hpp"

namespace vision {

PipelineResult run_pipeline(const Config& config, const std::atomic_bool& stop_requested) {
  InputSource source(config.source);
  source.open();
  Renderer renderer(config, source.fps());
  PipelineResult result;
  cv::Mat frame;
  while (!stop_requested.load()) {
    if (!source.next(frame)) break;
    ++result.frames;
    if (!renderer.render(frame)) break;
    if (config.max_frames && result.frames >= *config.max_frames) break;
  }
  result.interrupted = stop_requested.load();
  renderer.finalize();
  return result;
}

}  // namespace vision
