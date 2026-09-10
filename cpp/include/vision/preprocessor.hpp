#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include "vision/detection.hpp"
namespace vision {
struct ResizeMetadata { int original_height; int original_width; int resized_height; int resized_width; double ratio; };
struct PreprocessedFrame { std::vector<float> tensor; ResizeMetadata metadata; };
PreprocessedFrame preprocess(const cv::Mat& frame);
}  // namespace vision
