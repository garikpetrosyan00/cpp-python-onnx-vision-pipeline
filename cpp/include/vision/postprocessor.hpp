#pragma once
#include <string>
#include <vector>
#include "vision/detection.hpp"
#include "vision/preprocessor.hpp"
namespace vision {
constexpr int kModelHeight=416, kModelWidth=416, kClassCount=80, kOutputRows=3549, kOutputColumns=85;
void validate_output(const std::vector<float>& output);
float numpy_exp_float(float value);
std::vector<float> decode_boxes(const std::vector<float>& output);
float box_iou(const float* box, const float* other);
std::vector<std::size_t> nms(const std::vector<float>& boxes,const std::vector<float>& scores,float threshold);
std::vector<Detection> postprocess(const std::vector<float>& output,const ResizeMetadata& metadata,const std::vector<std::string>& labels,double confidence,double iou);
}  // namespace vision
