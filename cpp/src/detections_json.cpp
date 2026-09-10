#include "vision/detections_json.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <unistd.h>

namespace vision {
namespace {

std::string escape_json(const std::string& value) {
  std::ostringstream text;
  for (const unsigned char character : value) {
    if (character == '"' || character == '\\') {
      text << '\\' << static_cast<char>(character);
    } else if (character < 0x20) {
      text << "\\u" << std::hex << std::setw(4) << std::setfill('0')
           << static_cast<int>(character) << std::dec;
    } else {
      text << static_cast<char>(character);
    }
  }
  return text.str();
}

class TemporaryFile {
 public:
  explicit TemporaryFile(const std::filesystem::path& destination) {
    auto parent = destination.parent_path();
    if (parent.empty()) parent = std::filesystem::current_path();
    std::error_code error;
    std::filesystem::create_directories(parent, error);
    if (error) throw DetectorError("Cannot create detections JSON directory: " + error.message());
    std::string pattern = (parent / ("." + destination.filename().string() + ".XXXXXX.tmp")).string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int descriptor = mkstemps(writable.data(), 4);
    if (descriptor == -1) throw DetectorError("Cannot create temporary detections JSON beside " + destination.string());
    close(descriptor);
    path_ = writable.data();
  }
  ~TemporaryFile() {
    if (!published_) {
      std::error_code ignored;
      std::filesystem::remove(path_, ignored);
    }
  }
  const std::filesystem::path& path() const { return path_; }
  void publish(const std::filesystem::path& destination) {
    std::error_code error;
    std::filesystem::rename(path_, destination, error);
    if (error) throw DetectorError("Cannot publish detections JSON " + destination.string() + ": " + error.message());
    published_ = true;
  }

 private:
  std::filesystem::path path_;
  bool published_{false};
};

}  // namespace

void write_detections_json(const std::filesystem::path& path, const std::vector<Detection>& detections,
                           int height, int width, double confidence, double iou) {
  if (height <= 0 || width <= 0 || !std::isfinite(confidence) || !std::isfinite(iou))
    throw DetectorError("Cannot export invalid canonical detection metadata.");
  for (const auto& detection : detections) {
    if (!std::isfinite(detection.confidence) || !std::isfinite(detection.x1) ||
        !std::isfinite(detection.y1) || !std::isfinite(detection.x2) ||
        !std::isfinite(detection.y2))
      throw DetectorError("Canonical detections must contain finite numeric values.");
  }
  TemporaryFile temporary(path);
  std::ofstream output(temporary.path());
  if (!output) throw DetectorError("Cannot write detections JSON " + path.string());
  output << std::setprecision(std::numeric_limits<float>::max_digits10)
         << "{\"schema_version\":\"vision-pipeline-detections/v1\",\"implementation\":\"cpp\","
            "\"model_contract\":{\"name\":\"audited-yolox-nano\",\"sha256\":"
            "\"c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d\"},"
         << "\"image\":{\"height\":" << height << ",\"width\":" << width
         << "},\"thresholds\":{\"confidence\":" << confidence << ",\"iou\":" << iou
         << "},\"detections\":[";
  for (std::size_t index = 0; index < detections.size(); ++index) {
    const auto& detection = detections[index];
    if (index) output << ',';
    output << "{\"class_id\":" << detection.class_id << ",\"label\":\""
           << escape_json(detection.label) << "\",\"confidence\":" << detection.confidence
           << ",\"x1\":" << detection.x1 << ",\"y1\":" << detection.y1
           << ",\"x2\":" << detection.x2 << ",\"y2\":" << detection.y2 << '}';
  }
  output << "]}\n";
  output.close();
  if (!output) throw DetectorError("Cannot finalize detections JSON " + path.string());
  temporary.publish(path);
}

}  // namespace vision
