cmake_minimum_required(VERSION 3.10)
project(colorToGrayscale CUDA CXX)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

add_executable(colorToGrayscale colorToGrayscale.cu)
target_link_libraries(colorToGrayscale ${OpenCV_LIBS})