#include "../include/cuda_color_converter.h"


CUDA_EXPORTS bool test_cudaGray(const char* input_path)
{
    std::string path(input_path);
    cv::Mat input = cv::imread(path, cv::IMREAD_GRAYSCALE), output;

    cudaGray(input, output);

    cv::Mat reference_img;

    if (input.size() == output.size())
    {
        cv::Mat compare;
        cv::bitwise_xor(output, reference_img, compare);
        if (cv::countNonZero(compare) == 0) return true;
        else return false;
    }
    else return false;
}