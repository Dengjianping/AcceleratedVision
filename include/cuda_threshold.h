#ifndef CUDA_THRESHOLD_H
#define CUDA_THRESHOLD_H

#include "accel_vision.h"

/** @brief simply binarize a gray image.

@param input a Mat image, you better input a gray image.
@param output the result  after binarization.
@param threshold_value range 0 ~ 255.
*/
CUDA_EXPORTS void cudaThreshold(const cv::Mat & input, uchar thresh, uchar max_value, cv::Mat & output);


#endif // !CUDA_THRESHOLD_H
