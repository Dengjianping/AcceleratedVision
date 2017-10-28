#ifndef CUDA_ARITHMETIC_LOGICAL_H
#define CUDA_ARITHMETIC_LOGICAL_H

#include "..\include\accel_vision.h"

/** @brief simply get the sum of a gray image all pixels.

@param input a Mat image, you better input a gray image.
@param sum_of_pixels the sum of all pixels.
*/
CUDA_EXPORTS void cudaADD(const cv::Mat & input1, const cv::Mat & input2, cv::Mat & output);


CUDA_EXPORTS void cudaSqrt(const cv::Mat & input, cv::Mat & output);


CUDA_EXPORTS void cudaAND(const cv::Mat & input1, const cv::Mat & input2, cv::Mat & output);

#endif