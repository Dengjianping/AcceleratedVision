# because cmake fully supports cuda until 3.8
cmake_minimum_required(VERSION 3.8)
project(accel_vision LANGUAGES CXX CUDA)
file(GLOB SOURCES "*.cu" "*.cpp" "*.h" "*.cuh")


if(WIN32)
    set(OPENCV_INCLUDE_DIR "C:/Program Files/opencv/build/include")
    set(OPENCV_LIB_DIR "C:/Program Files/opencv/build/x64/vc14/lib")
    set(DEPENDENCY_LIB "opencv_world320d.lib")
    
    # include_directories(${PROJECT_SOURCE_DIR}/include)
    include_directories(${OPENCV_INCLUDE_DIR})
    include_directories(${OPENCV_INCLUDE_DIR}/opencv)
    include_directories(${OPENCV_INCLUDE_DIR}/opencv2)
    
    link_directories(accel_vision ${OPENCV_LIB_DIR})
    # aux_source_directory(${PROJECT_SOURCE_DIR}/sources DIR_SRCS)

    add_library(accel_vision SHARED 
        ${PROJECT_SOURCE_DIR}/sources/cuda_arithmetic_logical.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_blur.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_color_converter.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_sobel.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_split.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_sum.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_threshold.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_transform.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_warpaffine.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_bordering.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_dilation.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_erode.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_lut.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_merge.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_laplace.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_log.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_adjust.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_antisotropy.cu
        ${PROJECT_SOURCE_DIR}/sources/cuda_gamma_correction.cu
    )
    target_link_libraries(accel_vision ${DEPENDENCY_LIB})
    
elseif()
    message("AcceleratedVision doesn't support Linux now, but will support linux in the future.")
endif()