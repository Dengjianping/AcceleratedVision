#ifndef CUDASUM_H
#define CUDASUM_H

#include "accel_vision.h"


/** @get sum of all pixels from a image

@param input a Mat image, you better input a gray image or single channel image.
@param sum_result store each channel result

*/
CUDA_EXPORTS void cudaSum(const cv::Mat & input,int sum_result[3]);


/** @get mean value from a image

@param input a Mat image, you better input a gray image.
@param sum_result store each channel result

*/
CUDA_EXPORTS void cudaMean(const cv::Mat & input, float mean_result[3]);


#endif // !CUDASUM_H
