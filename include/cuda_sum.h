#ifndef CUDASUM_H
#define CUDASUM_H

#include "accel_vision.h"


/** @brief simply get the sum of a gray image all pixels.

@param input a Mat image, you better input a gray image.
@param sum_of_pixels the sum of all pixels.
*/
CUDA_EXPORTS void cudaSum(const cv::Mat & input, float & sum_of_pixels);


/** @brief simply get the mean value of a gray image all pixels.

@param input a Mat image, you better input a gray image.
@param mean_value the mean value of a gray image.
*/
CUDA_EXPORTS void cudaSum(const cv::Mat & input, float & mean_value);


#endif // !CUDASUM_H
