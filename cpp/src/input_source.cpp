#include "vision/input_source.hpp"

#include <cmath>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

namespace vision {
namespace {

class OpenCVCapture final : public Capture {
 public:
  bool open(int index) override { return capture_.open(index); }
  bool open(const std::string& path) override { return capture_.open(path); }
  bool is_opened() const override { return capture_.isOpened(); }
  bool read(cv::Mat& frame) override { return capture_.read(frame); }
  double get(int property) const override { return capture_.get(property); }
  void release() noexcept override { capture_.release(); }

 private:
  cv::VideoCapture capture_;
};

std::unique_ptr<Capture> default_capture() { return std::make_unique<OpenCVCapture>(); }

}  // namespace

InputSource::InputSource(SourceSpec source, CaptureFactory factory)
    : source_(std::move(source)), factory_(std::move(factory)) {
  if (!factory_) factory_ = default_capture;
}

InputSource::~InputSource() { close(); }

void InputSource::open() {
  if (opened_) throw std::runtime_error("Input source is already open; use one pipeline at a time.");
  frames_ = expected_frames_ = 0;
  eof_ = false;
  fps_ = 30.0;
  if (source_.kind == SourceKind::image) {
    image_ = cv::imread(source_.path.string(), cv::IMREAD_COLOR);
    if (image_.empty()) {
      throw std::runtime_error("Cannot read image " + source_.path.string() +
                               ". Check permissions and that it is a valid supported image.");
    }
    opened_ = true;
    return;
  }
  capture_ = factory_();
  if (!capture_) throw std::runtime_error("Cannot create OpenCV capture for media source.");
  const bool open_ok = source_.kind == SourceKind::camera ? capture_->open(source_.camera_index)
                                                            : capture_->open(source_.path.string());
  if (!open_ok || !capture_->is_opened()) {
    close();
    if (source_.kind == SourceKind::camera) {
      throw std::runtime_error("Camera " + std::to_string(source_.camera_index) +
                               " is unavailable. Check index, device permissions, connection, and other apps.");
    }
    throw std::runtime_error("Cannot open video " + source_.path.string() +
                             ". Check permissions, codec support, or file integrity.");
  }
  const double reported_fps = capture_->get(cv::CAP_PROP_FPS);
  if (std::isfinite(reported_fps) && reported_fps > 0.0) fps_ = reported_fps;
  const double count = capture_->get(cv::CAP_PROP_FRAME_COUNT);
  if (source_.kind == SourceKind::video && std::isfinite(count) && count > 0.0) {
    expected_frames_ = static_cast<std::size_t>(count);
  }
  opened_ = true;
}

bool InputSource::next(cv::Mat& frame) {
  if (!opened_) throw std::runtime_error("Input source is closed; acquire frames while it is open.");
  if (eof_) return false;
  if (source_.kind == SourceKind::image) {
    frame = image_;
    eof_ = true;
    return true;
  }
  frame.release();
  const bool read_ok = capture_->read(frame);
  if (!read_ok || frame.empty()) {
    if (source_.kind == SourceKind::camera) {
      throw std::runtime_error("Cannot read a frame from camera " + std::to_string(source_.camera_index) +
                               ". Check its connection and whether it is in use.");
    }
    if (frames_ == 0) {
      throw std::runtime_error("Video " + source_.path.string() +
                               " is empty or has no decodable frames. Check codec support and file integrity.");
    }
    if (frames_ < expected_frames_) {
      throw std::runtime_error("Video " + source_.path.string() + " became unreadable after " +
                               std::to_string(frames_) + " frames; file may be truncated.");
    }
    eof_ = true;
    return false;
  }
  ++frames_;
  return true;
}

double InputSource::fps() const noexcept { return fps_; }

void InputSource::close() noexcept {
  if (capture_) capture_->release();
  capture_.reset();
  image_.release();
  opened_ = false;
}

}  // namespace vision
