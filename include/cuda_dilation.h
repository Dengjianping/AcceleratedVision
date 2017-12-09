#ifndef CUDA_DILATION_H
#define CUDA_DILATION_H

#include "accel_vision.h"


/** @brief dilate a gray image.

@param input a Mat image, you better input a gray image or single channel image.
@param kernel_size dilation mask size.
@param eroded_times the times to dilate a image, now unsupported.
@param output the result image.
*/
CUDA_EXPORTS void cudaDilation(const cv::Mat & input, int kernel_size, int eroded_times, cv::Mat & output);

#endif // !CUDA_DILATION_H
