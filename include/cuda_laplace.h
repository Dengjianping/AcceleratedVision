#ifndef CUDA_LAPLACE_H
#define CUDA_LAPLACE_H

#include "accel_vision.h"

/** @brief simply binarize a gray image.

@param input a Mat image, you better input a gray image.
@param output the result  after binarization.
@param threshold_value range 0 ~ 255.
*/
enum LAPLACE_TYPE { NORMAL, EXTENDED };

CUDA_EXPORTS void cudaLaplace(const cv::Mat & input, cv::Mat & output, LAPLACE_TYPE lp_type);


#endif // !CUDA_LAPLACE_H