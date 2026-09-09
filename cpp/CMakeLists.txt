cmake_minimum_required(VERSION 3.20)

project(vision_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)

add_library(vision_core
  src/config.cpp
  src/input_source.cpp
  src/renderer.cpp
  src/pipeline.cpp
)
target_include_directories(vision_core PUBLIC include)
target_link_libraries(vision_core PUBLIC ${OpenCV_LIBS})

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
endif()
