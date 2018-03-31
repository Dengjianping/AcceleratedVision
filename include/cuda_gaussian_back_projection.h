#ifndef CUDA_GAUSSIAN_BACK_PROJECTION_H
#define CUDA_GAUSSIAN_BACK_PROJECTION_H

#include "accel_vision.h"


/** @gaussian back projection way to find similar area

@param input a 3-ch Mat image
@param model a 3-ch Mat image, better use a pure color image, for instance, if you want to spot red place from input image,
you better use a red image as model, the result will be more accuracy.

@param the result

*/
CUDA_EXPORTS void cudaGaussianBackProjection(const cv::Mat & input, const cv::Mat & model, cv::Mat & output);


#endif // !CUDA_GAUSSIAN_BACK_PROJECTION_H