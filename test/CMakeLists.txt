# because cmake fully supports cuda until 3.8
cmake_minimum_required(VERSION 3.8)
project(test_accel_vision LANGUAGES CXX)
file(GLOB SOURCES "*.cpp" "*.h")


# include files path
set(CUDA_INCLUDE_DIR $ENV{CUDA_PATH}/include) # read environment variable
set(OPENCV_INCLUDE_DIR "C:/Program Files/opencv/build/include")

#lib files path
set(OPENCV_LIB_DIR "C:/Program Files/opencv/build/x64/vc14/lib")
# use a relative path here
# reference: https://cmake.org/pipermail/cmake/2009-November/032988.html
set(ACCEL_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build/Release)
# lib file names
set(DEPENDENCY_OPENCV_LIB "opencv_world320.lib")
set(DEPENDENCY_ACCEL_LIB ${ACCEL_LIB_DIR}/accel_vision.lib)

# include header files
include_directories($ENV{CUDA_PATH}/include)
include_directories(${OPENCV_INCLUDE_DIR})
include_directories(${OPENCV_INCLUDE_DIR}/opencv)
include_directories(${OPENCV_INCLUDE_DIR}/opencv2)

# link to dependcies lib
link_directories(${OPENCV_LIB_DIR} ${ACCEL_LIB_DIR})

link_libraries(${DEPENDENCY_OPENCV_LIB})
link_libraries(${DEPENDENCY_ACCEL_LIB})

add_library(test_accel_vision SHARED 
    ${PROJECT_SOURCE_DIR}/test_cudaThreshold.cpp
    ${PROJECT_SOURCE_DIR}/test_cudaGray.cpp
)
add_library(test_accel_vision ${DEPENDENCY_ACCEL_LIB})

# target_link_libraries(test_accel_vision ${DEPENDENCY_ACCEL_LIB})