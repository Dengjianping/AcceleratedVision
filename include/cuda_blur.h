#ifndef CUDA_BLUR_H
#define CUDA_BLUR_H

#include "accel_vision.h"


/** @brief simply get the sum of a gray image all pixels.

@param input a Mat image, you better input a gray image or single channel image.
@param sum_of_pixels the sum of all pixels.
*/
CUDA_EXPORTS void cudaGaussianBlur(const cv::Mat & input, cv::Mat & output, int radius, float theta);


CUDA_EXPORTS void cudaGaussianBlurRGB(const cv::Mat & input, cv::Mat & output, int radius, float theta);

#endif // !CUDA_BLUR_H
