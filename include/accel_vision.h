#ifndef ACCEL_VISION_H
#define ACCEL_VISION_H


#ifdef CUDA_EXPORTS
    #define CUDA_EXPORTS extern "C" _declspec(dllimport)
#else
    #define CUDA_EXPORTS extern "C" _declspec(dllexport)
#endif // CUDA_EXPORTS


#define CUDA_CALL(x) {const cudaError_t a = (x); if (a != cudaSuccess) \
    { std::cout << std::endl << "CUDA Error: " << cudaGetErrorString(a) << ", error number: " << a << std::endl; cudaDeviceReset(); assert(0);}}


#define THREAD_MULTIPLE 32

// std include files
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>


// cuda include files
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <cufft.h>


// opencv include files
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#endif // !ACCEL_VISION_H
