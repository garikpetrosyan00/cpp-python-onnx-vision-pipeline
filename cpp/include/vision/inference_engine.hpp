#pragma once
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
namespace vision {
class InferenceEngine {
 public: explicit InferenceEngine(const std::filesystem::path& model); ~InferenceEngine()=default; InferenceEngine(const InferenceEngine&)=delete; InferenceEngine& operator=(const InferenceEngine&)=delete;
  std::vector<float> run(const std::vector<float>& tensor) const; const std::string& observed_metadata() const noexcept{return observed_;}
 private: Ort::Env environment_; Ort::SessionOptions options_; std::unique_ptr<Ort::Session> session_; std::string input_name_; std::string output_name_; std::string observed_;
};
}  // namespace vision
