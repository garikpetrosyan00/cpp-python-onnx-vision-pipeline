#include "test_support.hpp"
#include "vision/postprocessor.hpp"
#include "vision/preprocessor.hpp"
#include "vision/renderer.hpp"
#include "vision/inference_engine.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

void run_detector_tests() {
  using test_support::check;
  cv::Mat frame(100,200,CV_8UC3,cv::Scalar(1,2,3));
  const auto prepared=vision::preprocess(frame);
  check(prepared.tensor.size()==static_cast<std::size_t>(3*416*416),"Preprocess tensor shape failed.");
  check(std::abs(prepared.metadata.ratio-2.08)<1e-12 && prepared.metadata.resized_height==208,"Top-left letterbox metadata failed.");
  check(prepared.tensor[0]==1.F && prepared.tensor[416*416]==2.F && prepared.tensor[2*416*416]==3.F,"BGR NCHW conversion failed.");
  const std::size_t count=static_cast<std::size_t>(vision::kOutputRows)*vision::kOutputColumns;
  std::vector<float> output(count,0.F); for(int row=0;row<vision::kOutputRows;++row) output[static_cast<std::size_t>(row)*vision::kOutputColumns+4]=1.F;
  output[4]=1.F; output[5]=.5F; output[6]=.5F;
  auto boxes=vision::decode_boxes(output); check(std::abs(boxes[0]+4.F)<.001F && std::abs(boxes[2]-4.F)<.001F,"Stride-eight decode failed.");
  std::vector<float> nms_boxes{0,0,10,10,0,0,10,10}; std::vector<float> scores{.5F,.5F}; auto kept=vision::nms(nms_boxes,scores,.45F); check(kept.size()==1&&kept[0]==0,"Stable NMS tie ordering failed.");
  std::vector<std::string> labels(80); for(int i=0;i<80;++i)labels[i]="class"+std::to_string(i);
  auto results=vision::postprocess(output,prepared.metadata,labels,.5,.45); check(!results.empty()&&results.front().class_id==0,"Inclusive confidence filtering failed.");
  const auto passthrough=vision::Renderer::annotate(frame,{}); check(passthrough.data==frame.data,"Empty annotation must preserve passthrough frame.");
}

int run_real_model_smoke(const std::string& model_path, const std::string& labels_path) {
  if (!std::filesystem::is_regular_file(model_path) || !std::filesystem::is_regular_file(labels_path)) {
    std::cerr << "SKIP: audited local model or labels are unavailable; no download was attempted.\n";
    return 77;
  }
  std::ifstream stream(labels_path); std::vector<std::string> labels; std::string line;
  while (std::getline(stream,line)) labels.push_back(line);
  cv::Mat image(240,320,CV_8UC3); for(int y=0;y<image.rows;++y) for(int x=0;x<image.cols;++x) image.at<cv::Vec3b>(y,x)={static_cast<unsigned char>(x),static_cast<unsigned char>(y),static_cast<unsigned char>(x+y)};
  vision::InferenceEngine engine(model_path); const auto prepared=vision::preprocess(image); const auto detections=vision::postprocess(engine.run(prepared.tensor),prepared.metadata,labels,.01,.45);
  if(detections.empty()) throw std::runtime_error("Real-model smoke produced no detections at confidence 0.01.");
  const cv::Mat annotated=vision::Renderer::annotate(image,detections); if(cv::norm(image,annotated,cv::NORM_INF)==0) throw std::runtime_error("Real-model smoke did not annotate detections.");
  std::cout << "Real-model smoke passed with " << detections.size() << " detections.\n"; return 0;
}
