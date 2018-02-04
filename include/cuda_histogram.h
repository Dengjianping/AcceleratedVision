#ifndef CUDA_HISTOGRAM_H
#define CUDA_HISTOGRAM_H

#include "accel_vision.h"


/** @brief simply get the sum of a gray image all pixels.

@param input a Mat image, you better input a gray image or single channel image.
@param sum_of_pixels the sum of all pixels.
*/
CUDA_EXPORTS void cudaHistogram(const cv::Mat & input, uint *hist);

#endif // !CUDA_HISTOGRAM_H
