#include "vision/renderer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace vision {
namespace {

class OpenCVGui final : public Gui {
 public:
  void show(const std::string& name, const cv::Mat& frame) override {
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, frame);
  }
  int wait_key(int delay_ms) override { return cv::waitKey(delay_ms); }
  void destroy(const std::string& name) noexcept override {
    try {
      cv::destroyWindow(name);
    } catch (const cv::Exception&) {
    }
  }
};

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

int output_fourcc(const std::filesystem::path& path) {
  const auto extension = lowercase(path.extension().string());
  if (extension == ".avi" || extension == ".mkv") return cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  if (extension == ".mp4" || extension == ".mov") return cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  if (extension == ".webm") return cv::VideoWriter::fourcc('V', 'P', '8', '0');
  throw std::runtime_error("Unsupported video output extension: " + path.extension().string());
}

std::filesystem::path create_temporary(const std::filesystem::path& output) {
  std::filesystem::path parent = output.parent_path();
  if (parent.empty()) parent = std::filesystem::current_path();
  std::error_code error;
  std::filesystem::create_directories(parent, error);
  if (error) throw std::runtime_error("Cannot create output directory: " + error.message());
  const std::string suffix = output.extension().string();
  std::string pattern = (parent / ("." + output.stem().string() + "-XXXXXX" + suffix)).string();
  std::vector<char> characters(pattern.begin(), pattern.end());
  characters.push_back('\0');
  const int descriptor = mkstemps(characters.data(), static_cast<int>(suffix.size()));
  if (descriptor == -1) throw std::runtime_error("Cannot create temporary output beside " + output.string());
  close(descriptor);
  return std::filesystem::path(characters.data());
}

}  // namespace

Renderer::Renderer(const Config& config, double fps, std::unique_ptr<Gui> gui)
    : config_(config), fps_(fps), gui_(std::move(gui)) {
  if (!config_.no_display && !gui_) {
    const char* display = std::getenv("DISPLAY");
    const char* wayland = std::getenv("WAYLAND_DISPLAY");
    if (display == nullptr && wayland == nullptr) {
      throw std::runtime_error("No display session found. Run with --no-display for headless use.");
    }
    gui_ = std::make_unique<OpenCVGui>();
  }
}

Renderer::~Renderer() {
  if (writer_) writer_->release();
  if (!finalized_ && !temporary_.empty()) {
    std::error_code error;
    std::filesystem::remove(temporary_, error);
  }
  if (window_open_ && gui_) gui_->destroy(window_name_);
}

cv::Mat Renderer::annotate(const cv::Mat& frame, const std::vector<Detection>& detections) {
  if (detections.empty()) return frame;
  cv::Mat canvas=frame.clone();
  for(const auto& detection:detections) {
    const cv::Scalar color(64+(detection.class_id*37+29)%192,64+(detection.class_id*67+83)%192,64+(detection.class_id*97+137)%192);
    const int x1=std::clamp(static_cast<int>(detection.x1),0,canvas.cols-1), y1=std::clamp(static_cast<int>(detection.y1),0,canvas.rows-1), x2=std::clamp(static_cast<int>(detection.x2),0,canvas.cols-1), y2=std::clamp(static_cast<int>(detection.y2),0,canvas.rows-1);
    cv::rectangle(canvas,{x1,y1},{x2,y2},color,2); const std::string text=detection.label+" "+cv::format("%.2f",detection.confidence); cv::putText(canvas,text,{x1,std::max(0,y1-4)},cv::FONT_HERSHEY_SIMPLEX,.5,color,1,cv::LINE_AA);
  }
  return canvas;
}

bool Renderer::render(const cv::Mat& frame) {
  if (frame.empty()) throw std::runtime_error("Cannot render an empty frame.");
  if (config_.output) save(frame);
  if (config_.no_display) return true;
  try {
    gui_->show(window_name_, frame);
    window_open_ = true;
    const bool image = config_.source.kind == SourceKind::image;
    const int delay = image ? 30 : std::max(1, std::min(1000, static_cast<int>(1000.0 / fps_)));
    do {
      const int key = gui_->wait_key(delay) & 0xff;
      if (key == 'q' || key == 'Q' || key == 27) return false;
      if (!image) return true;
    } while (true);
  } catch (const cv::Exception& exception) {
    throw std::runtime_error(std::string("Cannot display frames. Use --no-display or check GUI support: ") +
                             exception.what());
  }
}

void Renderer::save(const cv::Mat& frame) {
  const auto& output = *config_.output;
  if (temporary_.empty()) temporary_ = create_temporary(output);
  try {
    if (config_.source.kind == SourceKind::image) {
      if (!cv::imwrite(temporary_.string(), frame)) {
        throw std::runtime_error("Cannot write image output " + output.string());
      }
    } else {
      if (frame.cols % 2 != 0 || frame.rows % 2 != 0) {
        throw std::runtime_error("Video output requires even frame dimensions to avoid codec cropping.");
      }
      if (!writer_) {
        size_ = frame.size();
        writer_ = std::make_unique<cv::VideoWriter>(temporary_.string(), output_fourcc(output), fps_, size_);
        if (!writer_->isOpened()) {
          throw std::runtime_error("Cannot open video output " + output.string() +
                                   ". Try .avi (MJPG) or install an encoder.");
        }
      }
      if (frame.size() != size_) {
        throw std::runtime_error("Cannot save changing frame dimensions to " + output.string());
      }
      writer_->write(frame);
    }
    ++frames_;
  } catch (const cv::Exception& exception) {
    throw std::runtime_error("Cannot write output " + output.string() + ": " + exception.what());
  }
}

void Renderer::verify_output() {
  if (config_.source.kind == SourceKind::image) {
    const cv::Mat saved = cv::imread(temporary_.string(), cv::IMREAD_COLOR);
    if (saved.empty()) throw std::runtime_error("Saved image is unreadable: " + config_.output->string());
    return;
  }
  cv::VideoCapture capture(temporary_.string());
  std::size_t count = 0;
  cv::Mat frame;
  while (capture.read(frame)) {
    if (frame.empty() || frame.size() != size_) {
      throw std::runtime_error("Saved video has invalid frame dimensions: " + config_.output->string());
    }
    ++count;
  }
  if (count != frames_) {
    throw std::runtime_error("Video output verification failed: wrote " + std::to_string(frames_) +
                             " frames but decoded " + std::to_string(count) + ". Try .avi (MJPG).");
  }
}

void Renderer::finalize() {
  if (finalized_) return;
  if (!config_.output || frames_ == 0) {
    finalized_ = true;
    return;
  }
  if (writer_) writer_->release();
  verify_output();
  std::error_code error;
  std::filesystem::rename(temporary_, *config_.output, error);
  if (error) throw std::runtime_error("Cannot publish output " + config_.output->string() + ": " + error.message());
  finalized_ = true;
}

}  // namespace vision
