#include "test_support.hpp"
#include "vision/pipeline.hpp"
#include "vision/renderer.hpp"

#include <atomic>
#include <memory>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

namespace {

class FakeGui final : public vision::Gui {
 public:
  struct State {
    bool shown{false};
    bool destroyed{false};
  };

  FakeGui(int key, std::shared_ptr<State> state) : key_(key), state_(std::move(state)) {}
  void show(const std::string&, const cv::Mat&) override { state_->shown = true; }
  int wait_key(int) override { return key_; }
  void destroy(const std::string&) noexcept override { state_->destroyed = true; }

 private:
  int key_;
  std::shared_ptr<State> state_;
};

}  // namespace

void run_pipeline_tests() {
  using test_support::check;
  using test_support::TemporaryDirectory;

  TemporaryDirectory directory;
  const auto source_path = directory.path() / "source.png";
  const auto output_path = directory.path() / "nested" / "output.png";
  cv::Mat original(8, 10, CV_8UC3);
  for (int row = 0; row < original.rows; ++row) {
    for (int col = 0; col < original.cols; ++col) original.at<cv::Vec3b>(row, col) = {static_cast<unsigned char>(row), static_cast<unsigned char>(col), 90};
  }
  check(cv::imwrite(source_path.string(), original), "Could not create pipeline image fixture.");

  vision::Config config{{vision::SourceKind::image, -1, source_path}, output_path, true, 1U, 0.25, 0.45};
  std::atomic_bool stop{false};
  const auto result = vision::run_pipeline(config, stop);
  check(result.frames == 1 && !result.interrupted, "Headless image pipeline did not process one frame.");
  const cv::Mat saved = cv::imread(output_path.string(), cv::IMREAD_COLOR);
  check(!saved.empty() && cv::norm(saved, original, cv::NORM_INF) == 0, "Output image differs from passthrough input.");

  const auto video_source = directory.path() / "source.avi";
  const auto video_output = directory.path() / "nested" / "output.avi";
  cv::VideoWriter fixture_writer(video_source.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 15.0,
                                 original.size());
  check(fixture_writer.isOpened(), "MJPG is unavailable for the video fixture.");
  fixture_writer.write(original);
  fixture_writer.write(original);
  fixture_writer.write(original);
  fixture_writer.release();
  vision::Config video_config{{vision::SourceKind::video, -1, video_source}, video_output, true, 2U, 0.25, 0.45};
  const auto video_result = vision::run_pipeline(video_config, stop);
  check(video_result.frames == 2, "Video max-frame limit failed.");
  cv::VideoCapture verified_video(video_output.string());
  cv::Mat video_frame;
  std::size_t output_frames = 0;
  while (verified_video.read(video_frame)) ++output_frames;
  check(output_frames == 2, "Video output did not contain the requested number of frames.");

  vision::Config interactive{{vision::SourceKind::image, -1, source_path}, std::nullopt, false, std::nullopt, 0.25, 0.45};
  auto gui_state = std::make_shared<FakeGui::State>();
  auto gui = std::make_unique<FakeGui>('q', gui_state);
  {
    vision::Renderer renderer(interactive, 30.0, std::move(gui));
    check(!renderer.render(original), "Q should exit interactive rendering.");
  }
  check(gui_state->shown && gui_state->destroyed, "GUI lifecycle was not completed.");

  auto esc_state = std::make_shared<FakeGui::State>();
  vision::Renderer esc_renderer(interactive, 30.0, std::make_unique<FakeGui>(27, esc_state));
  check(!esc_renderer.render(original), "ESC should exit interactive rendering.");
}
