cmake_minimum_required(VERSION 2.6)

project(bytetrack)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/include/eigen3)
link_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# cudnn
include_directories(/data/cuda/cuda-10.2/cudnn/v8.0.4/include)
link_directories(/data/cuda/cuda-10.2/cudnn/v8.0.4/lib64)
# tensorrt
include_directories(/opt/tiger/demo/TensorRT-7.2.3.4/include)
link_directories(/opt/tiger/demo/TensorRT-7.2.3.4/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB My_Source_Files ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_executable(bytetrack ${My_Source_Files})
target_link_libraries(bytetrack nvinfer)
target_link_libraries(bytetrack cudart)
target_link_libraries(bytetrack ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

