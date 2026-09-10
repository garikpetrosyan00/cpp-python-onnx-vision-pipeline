#pragma once

#include <cstddef>
#include <string>
#include <stdexcept>

namespace vision {
class DetectorError : public std::runtime_error { public: using std::runtime_error::runtime_error; };
struct Detection { int class_id; std::string label; float confidence; float x1; float y1; float x2; float y2; std::size_t anchor; };
}  // namespace vision
