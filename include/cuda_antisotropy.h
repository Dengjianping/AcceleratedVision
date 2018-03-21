#ifndef CUDA_ANTISOTROPY_H
#define CUDA_ANTISOTROPY_H

#include "accel_vision.h"


/** @reduce image noise without removing significant parts of the image content,
     see reference. https://en.wikipedia.org/wiki/Anisotropic_diffusion

@param input a Mat image
@param lamdba should be range 0.0 - 1.0
@param k int type
@param iteration int type 
@param output the result
@param gamma the gamma coefficiency
*/
CUDA_EXPORTS void cudaAntisotropy(const cv::Mat & input, float lamdba, int k, int iteration, cv::Mat & output);


#endif // !CUDA_ANTISOTROPY_H