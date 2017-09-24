#ifndef THRESHOLD_H
#define THRESHOLD_H

#include "accel_vision.h"

/** @brief simply binarize a gray image.

@param input a Mat image, you better input a gray image.
@param output the result  after binarization.
@param threshold_value range 0 ~ 255.
*/
CUDA_EXPORTS void cudaThreshold(const cv::Mat & input, cv::Mat & output, float threshold_value);


#endif // !THRESHOLD_H
