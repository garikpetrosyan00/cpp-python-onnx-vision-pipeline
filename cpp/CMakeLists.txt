cmake_minimum_required(VERSION 3.20)

project(vision_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "Root of the locally installed ONNX Runtime CPU archive")
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "ONNXRUNTIME_ROOT is required. Run scripts/setup_cpp.sh, then configure with -DONNXRUNTIME_ROOT=$<SHELL_PATH:${CMAKE_SOURCE_DIR}/../third_party/onnxruntime>")
endif()
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS "${ONNXRUNTIME_ROOT}/include" NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "Invalid ONNXRUNTIME_ROOT='${ONNXRUNTIME_ROOT}'. Expected include/onnxruntime_cxx_api.h and lib/libonnxruntime.so. Run scripts/setup_cpp.sh.")
endif()

add_library(vision_core
  src/config.cpp
  src/input_source.cpp
  src/renderer.cpp
  src/pipeline.cpp
  src/preprocessor.cpp
  src/postprocessor.cpp
  src/inference_engine.cpp
  src/detections_json.cpp
  src/metrics.cpp
)
target_include_directories(vision_core PUBLIC include "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(vision_core PUBLIC ${OpenCV_LIBS} "${ONNXRUNTIME_LIBRARY}")
set_target_properties(vision_core PROPERTIES BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")

add_executable(vision_cpp src/main.cpp)
target_link_libraries(vision_cpp PRIVATE vision_core)

foreach(target vision_core vision_cpp)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endforeach()

include(CTest)
if(BUILD_TESTING)
  add_executable(vision_cpp_tests
    tests/test_main.cpp
    tests/test_config.cpp
    tests/test_input_source.cpp
    tests/test_pipeline.cpp
    tests/test_detector.cpp
    tests/test_detections_json.cpp
    tests/test_metrics.cpp
  )
  target_link_libraries(vision_cpp_tests PRIVATE vision_core)
  if(MSVC)
    target_compile_options(vision_cpp_tests PRIVATE /W4)
  else()
    target_compile_options(vision_cpp_tests PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  add_test(NAME vision_cpp_tests COMMAND vision_cpp_tests)
  add_test(NAME vision_cpp_help COMMAND vision_cpp --help)
  add_test(NAME vision_cpp_version COMMAND vision_cpp --version)
  add_test(NAME vision_cpp_no_source_help COMMAND vision_cpp --no-display)
  add_test(NAME vision_cpp_invalid_option COMMAND vision_cpp --not-an-option)
  set_tests_properties(vision_cpp_invalid_option PROPERTIES WILL_FAIL TRUE)
  add_test(NAME vision_cpp_real_model_smoke COMMAND vision_cpp_tests --real-smoke "${CMAKE_SOURCE_DIR}/../models/detector.onnx" "${CMAKE_SOURCE_DIR}/../models/classes.txt")
  set_tests_properties(vision_cpp_real_model_smoke PROPERTIES SKIP_RETURN_CODE 77)
endif()
