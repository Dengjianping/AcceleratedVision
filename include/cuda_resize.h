#ifndef CUDA_RESIZE_H
#define CUDA_RESIZE_H

#include "..\include\accel_vision.h"

/** @brief simply get the sum of a gray image all pixels.

@param input a Mat image, you better input a gray image.
@param sum_of_pixels the sum of all pixels.
*/
CUDA_EXPORTS void cudaResize(const cv::Mat & input, cv::Mat & output, float scale);

#endif