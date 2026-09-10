#include "vision/postprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <limits>
#include <cstdint>
#include <cstring>

namespace vision {
namespace { constexpr std::size_t kValues=static_cast<std::size_t>(kOutputRows)*kOutputColumns; }
void validate_output(const std::vector<float>& output) {
  if(output.size()!=kValues) throw DetectorError("Unexpected model output element count "+std::to_string(output.size())+"; expected [1,3549,85] float32.");
  for(std::size_t i=0;i<output.size();++i) { if(!std::isfinite(output[i])) throw DetectorError("Model output contains NaN or infinite values."); if(i%kOutputColumns>=4 && (output[i]<0 || output[i]>1)) throw DetectorError("Model objectness/class probabilities must be in [0,1]; no logits expected."); }
}
float numpy_exp_float(float value) {
  if (value >= 88.72283935546875F) return std::numeric_limits<float>::infinity();
  if (value <= -103.97208404541015625F) return 0.0F;
  constexpr float magic=0x1.800000p+23F,log2e=1.442695040888963407359924681001892137F;
  volatile float rounded=value*log2e+magic; const float quadrant=rounded-magic;
  float x=std::fma(quadrant,-6.93145752e-1F,value); x=std::fma(quadrant,-1.42860677e-6F,x);
  float numerator=std::fma(5.082762527590693718096e-4F,x,6.757896990527504603057e-3F);
  numerator=std::fma(numerator,x,5.114512081637298353406e-2F); numerator=std::fma(numerator,x,2.473615434895520810817e-1F); numerator=std::fma(numerator,x,7.257664613233124478488e-1F); numerator=std::fma(numerator,x,1.0F);
  float denominator=std::fma(2.159509375685829852307e-2F,x,-2.742335390411667452936e-1F); denominator=std::fma(denominator,x,1.0F); float polynomial=numerator/denominator;
  std::uint32_t bits; std::memcpy(&bits,&polynomial,sizeof(bits)); const auto exponent_offset=static_cast<std::int64_t>(quadrant)*(std::int64_t{1}<<23); bits+=static_cast<std::uint32_t>(exponent_offset); std::memcpy(&polynomial,&bits,sizeof(bits)); return polynomial;
}
std::vector<float> decode_boxes(const std::vector<float>& output) {
  validate_output(output); std::vector<float> boxes(static_cast<std::size_t>(kOutputRows)*4); int row=0;
  for(int stride: {8,16,32}) for(int y=0;y<kModelHeight/stride;++y) for(int x=0;x<kModelWidth/stride;++x,++row) {
    const float* raw=&output[static_cast<std::size_t>(row)*kOutputColumns]; const double cx64=(static_cast<double>(raw[0])+x)*stride, cy64=(static_cast<double>(raw[1])+y)*stride; const float exp_w=numpy_exp_float(raw[2]),exp_h=numpy_exp_float(raw[3]); const double w64=static_cast<double>(exp_w)*stride,h64=static_cast<double>(exp_h)*stride; const float cx=static_cast<float>(cx64),cy=static_cast<float>(cy64),w=static_cast<float>(w64),h=static_cast<float>(h64);
    if(!std::isfinite(w)||!std::isfinite(h)) throw DetectorError("YOLOX box decoding overflowed; check the model/output contract.");
    float* box=&boxes[static_cast<std::size_t>(row)*4]; box[0]=cx-w/2; box[1]=cy-h/2; box[2]=cx+w/2; box[3]=cy+h/2;
  } return boxes;
}
float box_iou(const float* a,const float* b) { const float iw=std::max(0.F,std::min(a[2],b[2])-std::max(a[0],b[0])+1.F), ih=std::max(0.F,std::min(a[3],b[3])-std::max(a[1],b[1])+1.F); const float intersection=iw*ih; const float aa=std::max(0.F,a[2]-a[0]+1.F)*std::max(0.F,a[3]-a[1]+1.F); const float bb=std::max(0.F,b[2]-b[0]+1.F)*std::max(0.F,b[3]-b[1]+1.F); const float total=aa+bb-intersection; return total>0?intersection/total:0.F; }
std::vector<std::size_t> nms(const std::vector<float>& boxes,const std::vector<float>& scores,float threshold) { std::vector<std::size_t> order(scores.size()); std::iota(order.begin(),order.end(),0); std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return scores[a]>scores[b];}); std::vector<std::size_t> kept; for(auto candidate:order){ bool suppress=false; for(auto chosen:kept) if(box_iou(&boxes[candidate*4],&boxes[chosen*4])>threshold){suppress=true;break;} if(!suppress) kept.push_back(candidate); } return kept; }
std::vector<Detection> postprocess(const std::vector<float>& output,const ResizeMetadata& metadata,const std::vector<std::string>& labels,double confidence,double iou) {
  if(confidence<0||confidence>1||iou<0||iou>1||!std::isfinite(confidence)||!std::isfinite(iou)) throw DetectorError("Confidence and IoU must be finite values in [0,1].");
  if(labels.size()!=kClassCount||metadata.ratio<=0||!std::isfinite(metadata.ratio)||metadata.original_width<=0||metadata.original_height<=0) throw DetectorError("Invalid labels or coordinate restoration metadata for the 416x416 YOLOX contract.");
  auto boxes=decode_boxes(output); for(auto& value:boxes) value/=metadata.ratio; std::vector<Detection> detections;
  for(int class_id=0;class_id<kClassCount;++class_id){ std::vector<float> selected_boxes,scores; std::vector<std::size_t> anchors; for(int row=0;row<kOutputRows;++row){float score=output[static_cast<std::size_t>(row)*kOutputColumns+4]*output[static_cast<std::size_t>(row)*kOutputColumns+5+class_id]; if(score>=confidence){anchors.push_back(row);scores.push_back(score);selected_boxes.insert(selected_boxes.end(),boxes.begin()+row*4,boxes.begin()+row*4+4);}} for(auto selected:nms(selected_boxes,scores,static_cast<float>(iou))){auto anchor=anchors[selected]; const float* b=&boxes[anchor*4]; float x1=std::clamp(b[0],0.F,static_cast<float>(metadata.original_width)),y1=std::clamp(b[1],0.F,static_cast<float>(metadata.original_height)),x2=std::clamp(b[2],0.F,static_cast<float>(metadata.original_width)),y2=std::clamp(b[3],0.F,static_cast<float>(metadata.original_height)); if(x2>x1&&y2>y1) detections.push_back({class_id,labels[class_id],scores[selected],x1,y1,x2,y2,anchor}); }}
  std::sort(detections.begin(),detections.end(),[](const Detection&a,const Detection&b){if(a.confidence!=b.confidence)return a.confidence>b.confidence;if(a.class_id!=b.class_id)return a.class_id<b.class_id;return a.anchor<b.anchor;}); return detections;
}
}  // namespace vision
