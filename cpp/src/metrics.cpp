#include "vision/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <opencv2/core/version.hpp>

namespace vision {
namespace {
constexpr double kNanosecondsPerMillisecond = 1'000'000.0;
constexpr double kNanosecondsPerSecond = 1'000'000'000.0;

std::string json_number(const std::optional<double>& value) {
  if (!value) return "null";
  std::ostringstream text;
  text << std::setprecision(17) << *value;
  return text.str();
}

std::string json_escape(const std::string& value) {
  std::ostringstream escaped;
  for (const unsigned char character : value) {
    switch (character) {
      case '"': escaped << "\\\""; break;
      case '\\': escaped << "\\\\"; break;
      case '\b': escaped << "\\b"; break;
      case '\f': escaped << "\\f"; break;
      case '\n': escaped << "\\n"; break;
      case '\r': escaped << "\\r"; break;
      case '\t': escaped << "\\t"; break;
      default:
        if (character < 0x20) {
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << static_cast<unsigned int>(character) << std::dec << std::setfill(' ');
        } else {
          escaped << static_cast<char>(character);
        }
    }
  }
  return escaped.str();
}

std::string summary_json(const TimingSummary& summary) {
  std::ostringstream text;
  text << "{\"count\":" << summary.count << ",\"mean_ms\":" << json_number(summary.mean_ms)
       << ",\"median_ms\":" << json_number(summary.median_ms) << ",\"p50_ms\":"
       << json_number(summary.p50_ms) << ",\"p95_ms\":" << json_number(summary.p95_ms)
       << ",\"p99_ms\":" << json_number(summary.p99_ms) << ",\"min_ms\":"
       << json_number(summary.min_ms) << ",\"max_ms\":" << json_number(summary.max_ms) << '}';
  return text.str();
}

std::optional<std::int64_t> rss_bytes() {
  std::ifstream stream("/proc/self/statm");
  std::int64_t pages{};
  if (!(stream >> pages >> pages)) return std::nullopt;
  return pages * 4096;
}

std::string source_kind(SourceKind kind) {
  if (kind == SourceKind::image) return "image";
  if (kind == SourceKind::video) return "video";
  return "camera";
}

void write_file(const std::filesystem::path& path, const std::string& contents) {
  std::ofstream stream(path);
  if (!stream) throw std::runtime_error("Cannot stage benchmark result " + path.string());
  stream << contents;
  if (!stream) throw std::runtime_error("Cannot finalize benchmark result " + path.string());
}

std::filesystem::path unique_path(const std::filesystem::path& parent, const std::string& prefix) {
  std::string pattern = (parent / (prefix + "-XXXXXX.tmp")).string();
  std::vector<char> characters(pattern.begin(), pattern.end());
  characters.push_back('\0');
  const int descriptor = mkstemps(characters.data(), 4);
  if (descriptor == -1) throw std::runtime_error("Cannot create temporary benchmark result beside " + parent.string());
  close(descriptor);
  return characters.data();
}

void remove_if_present(const std::filesystem::path& path) noexcept {
  if (path.empty()) return;
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
}

bool exists_regular(const std::filesystem::path& path) {
  std::error_code error;
  const bool exists = std::filesystem::exists(path, error);
  if (error) throw std::runtime_error("Cannot access benchmark destination " + path.string() + ": " + error.message());
  if (exists && !std::filesystem::is_regular_file(path, error)) {
    throw std::runtime_error("Benchmark destination is not a regular file: " + path.string());
  }
  if (error) throw std::runtime_error("Cannot access benchmark destination " + path.string() + ": " + error.message());
  return exists;
}

}  // namespace

double percentile(const std::vector<std::int64_t>& samples, int percent) {
  if (samples.empty() || percent < 0 || percent > 100) throw std::runtime_error("Invalid percentile samples.");
  std::vector<std::int64_t> ordered = samples;
  std::sort(ordered.begin(), ordered.end());
  const double position = (ordered.size() - 1) * percent / 100.0;
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower);
}

TimingSummary timing_summary(const std::vector<std::int64_t>& samples) {
  if (samples.empty()) return {};
  const auto total = std::accumulate(samples.begin(), samples.end(), std::int64_t{0});
  const auto minimum = *std::min_element(samples.begin(), samples.end());
  const auto maximum = *std::max_element(samples.begin(), samples.end());
  const auto p50 = percentile(samples, 50);
  return {samples.size(), total / static_cast<double>(samples.size()) / kNanosecondsPerMillisecond,
          p50 / kNanosecondsPerMillisecond, p50 / kNanosecondsPerMillisecond,
          percentile(samples, 95) / kNanosecondsPerMillisecond,
          percentile(samples, 99) / kNanosecondsPerMillisecond, minimum / kNanosecondsPerMillisecond,
          maximum / kNanosecondsPerMillisecond};
}

void MetricsCollector::add(FrameTiming timing) {
  if (timing.capture_ns < 0 || timing.preprocess_ns < 0 || timing.inference_ns < 0 ||
      timing.postprocess_ns < 0 || timing.render_ns < 0 ||
      timing.total_ns != timing.capture_ns + timing.preprocess_ns + timing.inference_ns +
                             timing.postprocess_ns + timing.render_ns)
    throw std::runtime_error("Invalid frame timing boundaries.");
  frames_.push_back(timing);
}

std::optional<double> MetricsCollector::fps() const {
  std::int64_t total{};
  for (const auto& frame : frames_) total += frame.total_ns;
  if (total == 0) return std::nullopt;
  return frames_.size() * kNanosecondsPerSecond / total;
}

TimingSummary MetricsCollector::summary(const std::string& name) const {
  std::vector<std::int64_t> values;
  values.reserve(frames_.size());
  for (const auto& frame : frames_) {
    if (name == "capture") values.push_back(frame.capture_ns);
    else if (name == "preprocess") values.push_back(frame.preprocess_ns);
    else if (name == "inference") values.push_back(frame.inference_ns);
    else if (name == "postprocess") values.push_back(frame.postprocess_ns);
    else if (name == "render") values.push_back(frame.render_ns);
    else values.push_back(frame.total_ns);
  }
  return timing_summary(values);
}

std::filesystem::path default_benchmark_output() { return "benchmarks/results/cpp-benchmark.json"; }

void write_benchmark_results(const std::filesystem::path& destination, const Config& config,
                             const MetricsCollector& collector, std::size_t completed_warmup) {
  if (destination.extension() != ".json") throw std::runtime_error("--benchmark-output must be a .json path.");
  auto parent = destination.parent_path();
  if (parent.empty()) parent = std::filesystem::current_path();
  std::filesystem::create_directories(parent);
  const auto json = parent / destination.filename();
  const auto csv = parent / (destination.stem().string() + ".csv");
  if (json == csv) throw std::runtime_error("JSON and CSV benchmark destinations must be distinct.");
  const bool json_exists = exists_regular(json);
  const bool csv_exists = exists_regular(csv);
  const auto json_temp = unique_path(parent, "." + json.filename().string());
  const auto csv_temp = unique_path(parent, "." + csv.filename().string());
  std::filesystem::path json_backup;
  std::filesystem::path csv_backup;
  bool json_published = false;
  bool csv_published = false;
  const auto rss = rss_bytes();
  const auto capture = collector.summary("capture");
  const auto preprocess = collector.summary("preprocess");
  const auto inference = collector.summary("inference");
  const auto postprocess = collector.summary("postprocess");
  const auto render = collector.summary("render");
  const auto total = collector.summary("total");
  std::ostringstream document;
  document << "{\"schema_version\":\"1.0\",\"implementation\":\"cpp\",\"mode\":\""
           << (config.model ? "detector" : "passthrough") << "\",\"source_kind\":\""
           << source_kind(config.source.kind) << "\",\"requested_warmup_frames\":" << config.warmup
           << ",\"completed_warmup_frames\":" << completed_warmup << ",\"requested_measured_frames\":";
  if (config.max_frames) document << *config.max_frames; else document << "null";
  document << ",\"completed_measured_frames\":" << collector.frames().size() << ",\"confidence\":"
           << config.confidence << ",\"iou\":" << config.iou << ",\"no_display\":true,\"model_path\":";
  if (config.model) document << '"' << json_escape(config.model->string()) << '"'; else document << "null";
  document << ",\"model_sha256\":" << (config.model ? "\"c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d\"" : "null")
           << ",\"provider\":" << (config.model ? "\"CPUExecutionProvider\"" : "null")
           << ",\"versions\":{\"cpp\":\"C++17\",\"opencv\":\"" << CV_VERSION
           << "\",\"onnxruntime\":" << (config.model ? "\"1.24.4\"" : "null") << "}"
           << ",\"approximate_process_rss\":{\"bytes\":";
  if (rss) document << *rss; else document << "null";
  document << ",\"megabytes\":";
  if (rss) document << *rss / 1'000'000.0; else document << "null";
  document << ",\"meaning\":\"current process RSS snapshot after measured frames; not peak memory\"}"
           << ",\"timing_boundaries\":{\"unit\":\"milliseconds\",\"capture\":\"InputSource next(frame) only\",\"preprocess\":\"BGR frame to model tensor; zero in passthrough\",\"inference\":\"InferenceEngine.run only; zero in passthrough\",\"postprocess\":\"decode, score filter, class-aware NMS, restore, clamp; zero in passthrough\",\"render\":\"annotation, output write, and GUI draw; excludes waitKey delay\",\"total\":\"sum of capture, preprocess, inference, postprocess, render; excludes GUI wait\"}"
           << ",\"percentile_method\":\"linear interpolation at sorted index (n - 1) * p / 100\",\"timings\":{\"capture\":" << summary_json(capture) << ",\"preprocess\":" << summary_json(preprocess) << ",\"inference\":" << summary_json(inference) << ",\"postprocess\":" << summary_json(postprocess) << ",\"render\":" << summary_json(render) << ",\"total\":" << summary_json(total) << "},\"effective_fps\":" << json_number(collector.fps()) << "}\n";
  std::ostringstream row;
  row << "Implementation,Mode,Frames,WarmupFrames,CaptureMeanMs,PreprocessMeanMs,InferenceMeanMs,InferenceP50Ms,InferenceP95Ms,InferenceP99Ms,PostprocessMeanMs,RenderMeanMs,TotalMeanMs,TotalP50Ms,TotalP95Ms,TotalP99Ms,FPS,RSS_MB\n"
      << "cpp," << (config.model ? "detector" : "passthrough") << ',' << collector.frames().size() << ',' << completed_warmup << ','
      << json_number(capture.mean_ms) << ',' << json_number(preprocess.mean_ms) << ',' << json_number(inference.mean_ms) << ',' << json_number(inference.p50_ms) << ',' << json_number(inference.p95_ms) << ',' << json_number(inference.p99_ms) << ',' << json_number(postprocess.mean_ms) << ',' << json_number(render.mean_ms) << ',' << json_number(total.mean_ms) << ',' << json_number(total.p50_ms) << ',' << json_number(total.p95_ms) << ',' << json_number(total.p99_ms) << ',' << json_number(collector.fps()) << ',' << (rss ? std::to_string(*rss / 1'000'000.0) : "") << '\n';
  try {
    write_file(json_temp, document.str());
    write_file(csv_temp, row.str());
    if (json_exists) {
      json_backup = unique_path(parent, "." + json.filename().string() + ".backup");
      remove_if_present(json_backup);
      std::filesystem::rename(json, json_backup);
    }
    if (csv_exists) {
      csv_backup = unique_path(parent, "." + csv.filename().string() + ".backup");
      remove_if_present(csv_backup);
      std::filesystem::rename(csv, csv_backup);
    }
    std::filesystem::rename(json_temp, json);
    json_published = true;
    std::filesystem::rename(csv_temp, csv);
    csv_published = true;
    remove_if_present(json_backup);
    remove_if_present(csv_backup);
  } catch (...) {
    if (json_published) remove_if_present(json);
    if (csv_published) remove_if_present(csv);
    std::error_code ignored;
    if (!json_backup.empty()) std::filesystem::rename(json_backup, json, ignored);
    ignored.clear();
    if (!csv_backup.empty()) std::filesystem::rename(csv_backup, csv, ignored);
    remove_if_present(json_temp);
    remove_if_present(csv_temp);
    remove_if_present(json_backup);
    remove_if_present(csv_backup);
    throw;
  }
}

}  // namespace vision
