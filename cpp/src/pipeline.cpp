#include "vision/pipeline.hpp"

#include <chrono>
#include <stdexcept>

#include "vision/input_source.hpp"
#include "vision/renderer.hpp"
#include "vision/inference_engine.hpp"
#include "vision/postprocessor.hpp"
#include "vision/preprocessor.hpp"
#include "vision/detections_json.hpp"
#include "vision/metrics.hpp"

namespace vision {

PipelineResult run_pipeline(const Config& config, const std::atomic_bool& stop_requested) {
  std::unique_ptr<InferenceEngine> detector;
  if (config.model) detector=std::make_unique<InferenceEngine>(*config.model);
  InputSource source(config.source);
  source.open();
  Renderer renderer(config, source.fps());
  PipelineResult result;
  MetricsCollector collector;
  std::size_t warmup_completed{};
  std::vector<Detection> canonical_detections;
  cv::Size canonical_size;
  cv::Mat frame;
  while (!stop_requested.load()) {
    const auto capture_started = std::chrono::steady_clock::now();
    if (!source.next(frame)) break;
    const auto capture_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - capture_started).count();
    cv::Mat rendered=frame;
    std::int64_t preprocess_ns{};
    std::int64_t inference_ns{};
    std::int64_t postprocess_ns{};
    if(detector) {
      auto started=std::chrono::steady_clock::now();
      const auto prepared=preprocess(frame);
      preprocess_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count();
      started=std::chrono::steady_clock::now();
      const auto output=detector->run(prepared.tensor);
      inference_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count();
      started=std::chrono::steady_clock::now();
      canonical_detections=postprocess(output,prepared.metadata,config.label_names,config.confidence,config.iou);
      postprocess_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count();
      if(config.detections_json) canonical_size=frame.size();
      rendered=Renderer::annotate(frame,canonical_detections);
    }
    const auto render_started=std::chrono::steady_clock::now();
    renderer.render_processing(rendered);
    const auto render_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-render_started).count();
    const FrameTiming timing{capture_ns,preprocess_ns,inference_ns,postprocess_ns,render_ns,
                             capture_ns+preprocess_ns+inference_ns+postprocess_ns+render_ns};
    const bool is_warmup=config.benchmark && warmup_completed<config.warmup;
    if(is_warmup) ++warmup_completed;
    else {
      ++result.frames;
      if(config.benchmark) collector.add(timing);
    }
    if (!renderer.wait_for_exit()) break;
    if (config.max_frames && result.frames >= *config.max_frames) break;
  }
  result.interrupted = stop_requested.load();
  renderer.finalize();
  if(config.benchmark && !result.interrupted) {
    if(collector.frames().empty()) throw std::runtime_error("Benchmark reached end-of-file before one measured frame. Use fewer --warmup frames or a longer source.");
    write_benchmark_results(config.benchmark_output.value_or(default_benchmark_output()),config,collector,warmup_completed);
  }
  if(config.detections_json && !result.interrupted) write_detections_json(*config.detections_json,canonical_detections,canonical_size.height,canonical_size.width,config.confidence,config.iou);
  return result;
}

}  // namespace vision
