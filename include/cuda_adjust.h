#ifndef CUDA_ADJUST_H
#define CUDA_ADJUST_H

#include "accel_vision.h"


/** @brief adjust image intensity values or colormap.
this algrithm just like Matlab function imadjust, check the reference:
https://cn.mathworks.com/help/images/ref/imadjust.html

@param input a Mat image.
@param output a Mat image.
@param low_in in range (0.0, 1.0)
@param high_in in range (0.0, 1.0)
@param low_out in range (0.0, 1.0)
@param high_out in range (0.0, 1.0)
*/
CUDA_EXPORTS void cudaAdjust(const cv::Mat & input, cv::Mat & output, float low_in = 0.0f, float high_in = 1.0f, float low_out = 0.0f, float high_out = 1.0f);


#endif // !CUDA_ADJUST_H
