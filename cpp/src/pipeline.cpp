#include "vision/pipeline.hpp"

#include <stdexcept>

#include "vision/input_source.hpp"
#include "vision/renderer.hpp"
#include "vision/inference_engine.hpp"
#include "vision/postprocessor.hpp"
#include "vision/preprocessor.hpp"
#include "vision/detections_json.hpp"

namespace vision {

PipelineResult run_pipeline(const Config& config, const std::atomic_bool& stop_requested) {
  std::unique_ptr<InferenceEngine> detector;
  if (config.model) detector=std::make_unique<InferenceEngine>(*config.model);
  InputSource source(config.source);
  source.open();
  Renderer renderer(config, source.fps());
  PipelineResult result;
  std::vector<Detection> canonical_detections;
  cv::Size canonical_size;
  cv::Mat frame;
  while (!stop_requested.load()) {
    if (!source.next(frame)) break;
    ++result.frames;
    cv::Mat rendered=frame;
    if(detector) { const auto prepared=preprocess(frame); const auto output=detector->run(prepared.tensor); canonical_detections=postprocess(output,prepared.metadata,config.label_names,config.confidence,config.iou); if(config.detections_json) canonical_size=frame.size(); rendered=Renderer::annotate(frame,canonical_detections); }
    if (!renderer.render(rendered)) break;
    if (config.max_frames && result.frames >= *config.max_frames) break;
  }
  result.interrupted = stop_requested.load();
  renderer.finalize();
  if(config.detections_json && !result.interrupted) write_detections_json(*config.detections_json,canonical_detections,canonical_size.height,canonical_size.width,config.confidence,config.iou);
  return result;
}

}  // namespace vision
