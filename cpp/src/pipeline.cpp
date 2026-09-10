#include "vision/pipeline.hpp"

#include <stdexcept>

#include "vision/input_source.hpp"
#include "vision/renderer.hpp"
#include "vision/inference_engine.hpp"
#include "vision/postprocessor.hpp"
#include "vision/preprocessor.hpp"

namespace vision {

PipelineResult run_pipeline(const Config& config, const std::atomic_bool& stop_requested) {
  std::unique_ptr<InferenceEngine> detector;
  if (config.model) detector=std::make_unique<InferenceEngine>(*config.model);
  InputSource source(config.source);
  source.open();
  Renderer renderer(config, source.fps());
  PipelineResult result;
  cv::Mat frame;
  while (!stop_requested.load()) {
    if (!source.next(frame)) break;
    ++result.frames;
    cv::Mat rendered=frame;
    if(detector) { const auto prepared=preprocess(frame); const auto output=detector->run(prepared.tensor); rendered=Renderer::annotate(frame,postprocess(output,prepared.metadata,config.label_names,config.confidence,config.iou)); }
    if (!renderer.render(rendered)) break;
    if (config.max_frames && result.frames >= *config.max_frames) break;
  }
  result.interrupted = stop_requested.load();
  renderer.finalize();
  return result;
}

}  // namespace vision
