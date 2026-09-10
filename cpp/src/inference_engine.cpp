#include "vision/inference_engine.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include "vision/postprocessor.hpp"

namespace vision {
namespace {
constexpr std::array<int64_t,4> kInputShape{1,3,416,416};
constexpr std::array<int64_t,3> kOutputShape{1,3549,85};
std::string shape_string(const std::vector<int64_t>& shape) { std::ostringstream text; text<<"["; for(std::size_t i=0;i<shape.size();++i){if(i)text<<",";text<<shape[i];} return text<<"]",text.str(); }
template <std::size_t N>
bool matches(const std::vector<int64_t>& actual,const std::array<int64_t,N>& expected) { return actual.size()==expected.size() && std::equal(actual.begin(),actual.end(),expected.begin()); }
}
InferenceEngine::InferenceEngine(const std::filesystem::path& model)
    : environment_(ORT_LOGGING_LEVEL_WARNING,"vision_cpp"), options_() {
  try {
    if(!std::filesystem::is_regular_file(model)) throw DetectorError("Model is missing or not a regular file: "+model.string());
    std::ifstream stream(model,std::ios::binary); if(!stream.good()) throw DetectorError("Cannot read model "+model.string()+". Check path and permissions.");
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_=std::make_unique<Ort::Session>(environment_,model.c_str(),options_);
    const auto inputs=session_->GetInputCount(),outputs=session_->GetOutputCount();
    std::ostringstream observed; observed<<"inputs="<<inputs<<", outputs="<<outputs;
    if(inputs==1){auto type_info=session_->GetInputTypeInfo(0); auto info=type_info.GetTensorTypeAndShapeInfo(); observed<<", input="<<session_->GetInputNameAllocated(0,Ort::AllocatorWithDefaultOptions()).get()<<" "<<shape_string(info.GetShape())<<" type="<<info.GetElementType();}
    if(outputs==1){auto type_info=session_->GetOutputTypeInfo(0); auto info=type_info.GetTensorTypeAndShapeInfo(); observed<<", output="<<session_->GetOutputNameAllocated(0,Ort::AllocatorWithDefaultOptions()).get()<<" "<<shape_string(info.GetShape())<<" type="<<info.GetElementType();}
    observed_ = observed.str();
    if(inputs!=1||outputs!=1) throw DetectorError("Incompatible model metadata: "+observed_+"; expected one input and one output.");
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_type_info=session_->GetInputTypeInfo(0); auto output_type_info=session_->GetOutputTypeInfo(0); auto input_info=input_type_info.GetTensorTypeAndShapeInfo(); auto output_info=output_type_info.GetTensorTypeAndShapeInfo();
    if(input_info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||output_info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||!matches(input_info.GetShape(),kInputShape)||!matches(output_info.GetShape(),kOutputShape)) throw DetectorError("Incompatible model metadata: "+observed_+"; expected float input [1,3,416,416] and float output [1,3549,85].");
    input_name_=session_->GetInputNameAllocated(0,allocator).get(); output_name_=session_->GetOutputNameAllocated(0,allocator).get();
  } catch(const Ort::Exception& error) { throw DetectorError("Model "+model.string()+"; CPUExecutionProvider; observed "+observed_+". "+error.what()); }
}
std::vector<float> InferenceEngine::run(const std::vector<float>& tensor) const {
  if(tensor.size()!=static_cast<std::size_t>(1*3*416*416)) throw DetectorError("Expected contiguous float32 input [1,3,416,416].");
  for(float value:tensor) if(!std::isfinite(value)) throw DetectorError("Expected finite float32 input [1,3,416,416].");
  const std::array<int64_t,4> shape{1,3,416,416}; auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
  auto input=Ort::Value::CreateTensor<float>(memory,const_cast<float*>(tensor.data()),tensor.size(),shape.data(),shape.size()); const char* in[]={input_name_.c_str()}; const char* out[]={output_name_.c_str()};
  try { auto values=session_->Run(Ort::RunOptions{nullptr},in,&input,1,out,1); if(values.size()!=1||!values[0].IsTensor()) throw DetectorError("Expected one tensor output from ONNX Runtime."); auto info=values[0].GetTensorTypeAndShapeInfo(); if(info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||!matches(info.GetShape(),kOutputShape)) throw DetectorError("Runtime output metadata changed: expected [1,3549,85] float32."); const float* data=values[0].GetTensorData<float>(); std::vector<float> result(data,data+kOutputRows*kOutputColumns); validate_output(result); return result; }
  catch(const Ort::Exception& error){throw DetectorError("ONNX Runtime CPU inference failed: "+std::string(error.what()));}
}
}  // namespace vision
