#include "../include/cuda_threshold.h"


CUDA_EXPORTS bool test_cudaThreshold(const char* input_path, uchar thresh, uchar max_value)
{
    std::string path(input_path);
    cv::Mat input = cv::imread(path, cv::IMREAD_GRAYSCALE), output;

    // calling test api
    cudaThreshold(input, thresh, max_value, output);

    cv::Mat reference_img;
    cv::threshold(input, reference_img, thresh, max_value, CV_8U);

    if (input.size() == output.size())
    {
        cv::Mat compare;
        cv::bitwise_xor(output, reference_img, compare);
        if (cv::countNonZero(compare) == 0) return true;
        else return false;
    }
    else return false;
}