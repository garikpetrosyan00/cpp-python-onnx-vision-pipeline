#pragma once
#include <filesystem>
#include <vector>
#include "vision/detection.hpp"
namespace vision { void write_detections_json(const std::filesystem::path& path,const std::vector<Detection>& detections,int height,int width,double confidence,double iou); }
