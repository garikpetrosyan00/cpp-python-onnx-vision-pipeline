#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "vision/config.hpp"

namespace vision {

class Gui {
 public:
  virtual ~Gui() = default;
  virtual void show(const std::string& name, const cv::Mat& frame) = 0;
  virtual int wait_key(int delay_ms) = 0;
  virtual void destroy(const std::string& name) noexcept = 0;
};

class Renderer {
 public:
  Renderer(const Config& config, double fps, std::unique_ptr<Gui> gui = {});
  ~Renderer();
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;

  bool render(const cv::Mat& frame);
  void finalize();

 private:
  void save(const cv::Mat& frame);
  void verify_output();
  const Config& config_;
  double fps_;
  std::unique_ptr<Gui> gui_;
  std::unique_ptr<cv::VideoWriter> writer_;
  std::filesystem::path temporary_;
  cv::Size size_;
  std::size_t frames_{0};
  bool window_open_{false};
  bool finalized_{false};
  const std::string window_name_{"Vision pipeline - Q / ESC to exit"};
};

}  // namespace vision
