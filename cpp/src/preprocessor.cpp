#include "vision/preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "vision/postprocessor.hpp"

namespace vision {
PreprocessedFrame preprocess(const cv::Mat& frame) {
  if (frame.empty() || frame.type() != CV_8UC3) throw DetectorError("Preprocessing expects a nonempty HxWx3 uint8 BGR frame.");
  const int height=frame.rows, width=frame.cols;
  const double ratio=std::min(static_cast<double>(kModelHeight)/height,static_cast<double>(kModelWidth)/width);
  const int resized_width=static_cast<int>(width*ratio), resized_height=static_cast<int>(height*ratio);
  if (resized_width<1 || resized_height<1) throw DetectorError("Frame is too narrow for the audited 416x416 resize.");
  cv::Mat resized, canvas(kModelHeight,kModelWidth,CV_8UC3,cv::Scalar(114,114,114));
  try { cv::resize(frame,resized,{resized_width,resized_height},0,0,cv::INTER_LINEAR); resized.copyTo(canvas(cv::Rect(0,0,resized_width,resized_height))); }
  catch(const cv::Exception& error){ throw DetectorError(std::string("Cannot resize frame for YOLOX-Nano: ")+error.what()); }
  std::vector<float> tensor(static_cast<std::size_t>(3)*kModelHeight*kModelWidth);
  const int plane=kModelHeight*kModelWidth;
  for(int y=0;y<kModelHeight;++y) for(int x=0;x<kModelWidth;++x) { const auto pixel=canvas.at<cv::Vec3b>(y,x); for(int c=0;c<3;++c) tensor[c*plane+y*kModelWidth+x]=pixel[c]; }
  return {std::move(tensor),{height,width,resized_height,resized_width,ratio}};
}
}  // namespace vision
