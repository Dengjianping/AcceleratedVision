#include "../include/cuda_bordering.h"


__global__ void bordering_rect(uchar *d_input, int height, int width, int top, int bottom, int left, int right, uchar color, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x; // each thread handle 4 pixels

    for (int i = row; i < height; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            if (i < top || height - i < bottom || j < left || width - j < right)
                d_output[i*width + j] = color; // top or bottom
            else
                d_output[i*width + j] = d_input[i*width + j];
        }
}


void cudaBordering(const cv::Mat & input, int top, int bottom, int left, int right, uchar color, cv::Mat & output)
{
    output = cv::Mat(input.size(), CV_8U, cv::Scalar(0));
    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 6);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (4 * block_size.x), input.rows / (4 * block_size.y));

    uchar *d_input, *d_output;
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&d_output, sizeof(uchar)*input.cols*input.rows));

    // calling kernel
    bordering_rect <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, top, bottom, left, right, color, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(output.data, d_output, sizeof(uchar)*output.cols*output.rows, cudaMemcpyDeviceToHost));

    // resources releasing
    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
}


__global__ void bordering_circle(uchar *d_input, int height, int width, float radius, uchar color, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x; // each thread handle 4 pixels

    for (int i = row; i < height; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            float x = j - width / 2, y = height / 2 - i;
            float circle = powf(x, 2.0f) + powf(y, 2.0f);
            float r2 = powf(radius, 2.0f);
            if (circle <= r2)
                d_output[i*width + j] = d_input[i*width + j];
            else
                d_output[i*width + j] = color;
        }
}


void cudaBorderingCircle(const cv::Mat & input, float radius, uchar color, cv::Mat & output)
{
    output = cv::Mat(input.size(), CV_8U, cv::Scalar(0));
    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 6);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (4 * block_size.x), input.rows / (4 * block_size.y));

    uchar *d_input, *d_output;
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&d_output, sizeof(uchar)*input.cols*input.rows));

    // calling kernel
    bordering_circle <<<grid_size, block_size, 0, stream >>> (d_input, input.rows, input.cols, radius, color, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(output.data, d_output, sizeof(uchar)*output.cols*output.rows, cudaMemcpyDeviceToHost));

    // resources releasing
    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
}


__global__ void bordering_ellipse(uchar *d_input, int height, int width, float radius_x, float radius_y, uchar color, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x; // each thread handle 4 pixels

    for (int i = row; i < height; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            float x = j - width / 2, y = height / 2 - i;
            float x_2 = powf(x, 2.0f);
            float y_2 = powf(y, 2.0f);
            float rx_2 = powf(radius_x, 2.0f);
            float ry_2 = powf(radius_y, 2.0f);
            float ellipse = x_2 / rx_2 + y_2 / ry_2;
            if (ellipse <= 1.0f)
                d_output[i*width + j] = d_input[i*width + j];
            else
                d_output[i*width + j] = color;
        }
}


void cudaBorderingEllipse(const cv::Mat & input, float radius_x, float radius_y, uchar color, cv::Mat & output)
{
    output = cv::Mat(input.size(), CV_8U, cv::Scalar(0));
    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 6);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (4 * block_size.x), input.rows / (4 * block_size.y));

    uchar *d_input, *d_output;
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&d_output, sizeof(uchar)*input.cols*input.rows));

    // calling kernel
    bordering_ellipse <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, radius_x, radius_y, color, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(output.data, d_output, sizeof(uchar)*output.cols*output.rows, cudaMemcpyDeviceToHost));

    // resources releasing
    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
}