#ifndef CUDA_MERGE_H
#define CUDA_MERGE_H

#include "..\include\accel_vision.h"

/** @brief simply get the sum of a gray image all pixels.

@param input a Mat image, you better input a gray image.
@param sum_of_pixels the sum of all pixels.
*/
CUDA_EXPORTS void cudaMerge(const std::vector<cv::Mat> & channels, cv::Mat & output);

#endif