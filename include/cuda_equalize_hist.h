#ifndef CUDA_EQUALIZEHIST_H
#define CUDA_EQUALIZEHIST_H

#include "accel_vision.h"


/** @brief simply get the sum of a gray image all pixels.

@param input a Mat image, you better input a gray image or single channel image.
@param sum_of_pixels the sum of all pixels.
*/
CUDA_EXPORTS void cudaEqualizeHist(const cv::Mat & input, cv::Mat & output);

#endif // !CUDA_EQUALIZEHIST_H
