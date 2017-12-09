#ifndef CUDA_ERODE_H
#define CUDA_ERODE_H

#include "accel_vision.h"


/** @brief erode a gray image.

@param input a Mat image, you better input a gray image or single channel image.
@param kernel_size erode mask size.
@param eroded_times the times to dilate a image, now unsupported.
@param output the result image.
*/
CUDA_EXPORTS void cudaErode(const cv::Mat & input, int kernel_size, int eroded_times, cv::Mat & output);

#endif // !CUDA_ERODE_H
