#include "test_support.hpp"
#include "vision/input_source.hpp"

#include <deque>
#include <memory>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

namespace {

class FakeCapture final : public vision::Capture {
 public:
  bool open(int) override { return opens; }
  bool open(const std::string&) override { return opens; }
  bool is_opened() const override { return opens; }
  bool read(cv::Mat& frame) override {
    if (frames.empty()) return false;
    frame = frames.front();
    frames.pop_front();
    return true;
  }
  double get(int property) const override { return property == cv::CAP_PROP_FPS ? 24.0 : frame_count; }
  void release() noexcept override { *released = true; }

  bool opens{true};
  std::shared_ptr<bool> released{std::make_shared<bool>(false)};
  double frame_count{0};
  std::deque<cv::Mat> frames;
};

}  // namespace

void run_input_source_tests() {
  using test_support::check;
  using test_support::expect_throw;
  using test_support::TemporaryDirectory;

  TemporaryDirectory directory;
  const auto image_path = directory.path() / "fixture.png";
  const cv::Mat expected(4, 6, CV_8UC3, cv::Scalar(10, 20, 30));
  check(cv::imwrite(image_path.string(), expected), "Could not create image fixture.");
  vision::InputSource image({vision::SourceKind::image, -1, image_path});
  image.open();
  cv::Mat frame;
  check(image.next(frame) && frame.size() == expected.size(), "Image frame acquisition failed.");
  check(!image.next(frame), "Image source must reach EOF after one frame.");
  image.close();

  std::shared_ptr<bool> released;
  vision::InputSource video(
      {vision::SourceKind::video, -1, directory.path() / "fixture.avi"}, [&] {
        auto capture = std::make_unique<FakeCapture>();
        capture->frame_count = 1;
        capture->frames.push_back(expected);
        released = capture->released;
        return capture;
      });
  video.open();
  check(video.fps() == 24.0 && video.next(frame), "Video capture acquisition failed.");
  check(!video.next(frame), "Video EOF handling failed.");
  video.close();
  check(*released, "Capture was not released.");

  expect_throw(
      [&] {
        vision::InputSource empty_video({vision::SourceKind::video, -1, directory.path() / "empty.avi"}, [] {
          return std::make_unique<FakeCapture>();
        });
        empty_video.open();
        cv::Mat empty_frame;
        empty_video.next(empty_frame);
      },
      "empty or has no decodable frames");

  expect_throw(
      [&] {
        vision::InputSource unavailable({vision::SourceKind::camera, 0, {}}, [] {
          auto capture = std::make_unique<FakeCapture>();
          capture->opens = false;
          return capture;
        });
        unavailable.open();
      },
      "Camera 0 is unavailable");

  std::shared_ptr<bool> camera_released;
  expect_throw(
      [&] {
        vision::InputSource failing_camera({vision::SourceKind::camera, 2, {}}, [&] {
          auto capture = std::make_unique<FakeCapture>();
          camera_released = capture->released;
          return capture;
        });
        failing_camera.open();
        cv::Mat camera_frame;
        failing_camera.next(camera_frame);
      },
      "Cannot read a frame from camera 2");
  check(*camera_released, "Capture was not released after camera read failure.");
}
