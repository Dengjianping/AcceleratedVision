#ifndef CUDA_BORDERING_H
#define CUDA_BORDERING_H

#include "accel_vision.h"


/** @brief simply get the sum of a gray image all pixels.

@param input a Mat image, you better input a gray image or single channel image.
@param sum_of_pixels the sum of all pixels.
*/
CUDA_EXPORTS void cudaBordering(const cv::Mat & input, int top, int bottom, int left, int right, uchar color, cv::Mat & output);

CUDA_EXPORTS void cudaBorderingCircle(const cv::Mat & input, float radius, uchar color, cv::Mat & output);

CUDA_EXPORTS void cudaBorderingEllipse(const cv::Mat & input, float radius_x, float radius_y, uchar color, cv::Mat & output);

#endif // !CUDA_BORDERING_H
