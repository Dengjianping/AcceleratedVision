#ifndef CUDA_WARPAFFINE_H
#define CUDA_WARPAFFINE_H

#include "accel_vision.h"

/** @brief simply binarize a gray image.

@param input a Mat image, you better input a gray image.
@param output the result  after binarization.
@param threshold_value range 0 ~ 255.
*/
CUDA_EXPORTS void cudaWarpAffine(const cv::Mat & input, cv::Mat & output, float degree);


#endif // !CUDA_WARPAFFINE_H
