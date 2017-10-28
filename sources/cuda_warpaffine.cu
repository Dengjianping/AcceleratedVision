#include "../include/cuda_warpaffine.h"


__global__ void warpaffine(uchar *d_input, int height, int width, float degree, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x; // each thread handle 4 pixels

    auto coor = [](const int & x, const int & y, const int & height, const int & width, const float & degree)
    {
        float m11 = cosf(degree);
        float m21 = sinf(degree);
        float m13 = (float)width / 2 * (1 - m11) - m21*(float)height / 2;
        float m23 = (float)width / 2 * m21 + (1 - m11)*(float)height / 2;
        int new_x = m11*x + m21*y + m13;
        int new_y = m11*y - m21*x + m23;
        return make_int2(new_y, new_x);
    };

    auto in_range = [](const int & x, const int & y, int height, int width)
    {
        return true;
    };

    if (row < height && col < width)
    {
        int x = col - width / 2, y = height / 2 - row; // each pixel coordinator
        int2 point = coor(x, y, height, width, degree);
        int new_x = point.x, new_y = point.y;
        int new_row = height / 2 - new_y, new_col = new_x + width / 2;
        if (new_row > height || new_col > width)
        {

            d_output[row*width + col] = 0;
        }
        else
        {
            d_output[new_row*width + new_col] = d_input[row*width + col];

        }
    }
}


void cudaWarpAffine(const cv::Mat & input, cv::Mat & output, float degree)
{
    degree = PI*degree / 180.0f;
    output = cv::Mat(input.size(), input.type(), cv::Scalar(0));

    cudaStream_t stream; CUDA_CALL(cudaStreamCreate(&stream));

    uchar *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(uchar)*input.rows*input.cols);
    cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.rows*input.cols, cudaMemcpyHostToDevice, stream);
    cudaMalloc(&d_output, sizeof(uchar)*input.rows*input.cols);

    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 6);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (1 * block_size.x), input.rows / (1 * block_size.y));

    warpaffine <<<grid_size, block_size, 0, stream>>>(d_input, input.rows, input.cols, degree, d_output);
    cudaDeviceSynchronize();

    cudaMemcpyAsync(output.data, d_output, sizeof(uchar)*input.rows*input.cols, cudaMemcpyDeviceToHost, stream);
    cudaFree(d_input); cudaFree(d_output); cudaStreamDestroy(stream);
}