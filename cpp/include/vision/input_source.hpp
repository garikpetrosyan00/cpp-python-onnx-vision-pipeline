#pragma once

#include <functional>
#include <memory>

#include <opencv2/core.hpp>

#include "vision/config.hpp"

namespace vision {

class Capture {
 public:
  virtual ~Capture() = default;
  virtual bool open(int index) = 0;
  virtual bool open(const std::string& path) = 0;
  virtual bool is_opened() const = 0;
  virtual bool read(cv::Mat& frame) = 0;
  virtual double get(int property) const = 0;
  virtual void release() noexcept = 0;
};

using CaptureFactory = std::function<std::unique_ptr<Capture>()>;

class InputSource {
 public:
  explicit InputSource(SourceSpec source, CaptureFactory factory = {});
  ~InputSource();
  InputSource(const InputSource&) = delete;
  InputSource& operator=(const InputSource&) = delete;
  InputSource(InputSource&&) noexcept = default;
  InputSource& operator=(InputSource&&) noexcept = default;

  void open();
  bool next(cv::Mat& frame);
  double fps() const noexcept;
  void close() noexcept;

 private:
  SourceSpec source_;
  CaptureFactory factory_;
  std::unique_ptr<Capture> capture_;
  cv::Mat image_;
  bool opened_{false};
  bool eof_{false};
  std::size_t frames_{0};
  std::size_t expected_frames_{0};
  double fps_{30.0};
};

}  // namespace vision
