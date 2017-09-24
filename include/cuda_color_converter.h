#ifndef CUDA_GRAY_H
#define CUDA_GRAY_H


#include "accel_vision.h"

/** @brief simply get the sum of a gray image all pixels.

@param input a Mat image, you better input a gray image.
@param sum_of_pixels the sum of all pixels.
*/
CUDA_EXPORTS void cudaGray(const cv::Mat & input, cv::Mat & output);


/** @brief simply get the sum of a gray image all pixels.

@param input a Mat image, you better input a gray image.
@param sum_of_pixels the sum of all pixels.
*/
CUDA_EXPORTS void cudaHSV(const cv::Mat & input, cv::Mat & output);


#endif