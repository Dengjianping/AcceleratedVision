#ifndef CUDASUM_H
#define CUDASUM_H

#include "accel_vision.h"


/** @get sum of all pixels from a image

@param input a Mat image, you better input a gray image or single channel image.

*/
CUDA_EXPORTS int cudaSum(const cv::Mat & input);


/** @get mean value from a image

@param input a Mat image, you better input a gray image.

*/
CUDA_EXPORTS float cudaMean(const cv::Mat & input);


#endif // !CUDASUM_H
