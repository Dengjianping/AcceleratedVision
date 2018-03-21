#ifndef CUDA_GAMMA_CORRECTION_H
#define CUDA_GAMMA_CORRECTION_H

#include "accel_vision.h"


/** @Increases or decreases gamma value of input image.

@param input a Mat image
@param output the result
@param gamma the gamma coefficiency
*/
CUDA_EXPORTS void cudaGammaCorrection(const cv::Mat & input, cv::Mat & output, float gamma=1.0f);


#endif // !CUDA_GAMMA_CORRECTION_H
