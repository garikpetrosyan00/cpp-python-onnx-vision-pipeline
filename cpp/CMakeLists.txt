cmake_minimum_required(VERSION 3.20)

project(vision_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)

add_executable(vision_cpp src/main.cpp)
target_include_directories(vision_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision_cpp PRIVATE ${OpenCV_LIBS})

if(MSVC)
  target_compile_options(vision_cpp PRIVATE /W4)
else()
  target_compile_options(vision_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME vision_cpp_help COMMAND vision_cpp --help)
  add_test(NAME vision_cpp_version COMMAND vision_cpp --version)
endif()
