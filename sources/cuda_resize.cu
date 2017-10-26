#include "../include/cuda_resize.h"


// Bilinear Interpolation
__device__ float bilinear(float q11, float q12, float q21, float q22, float scale)
{
    return fabsf(1.0f - scale)*fabsf(1.0f - scale)*q11 + fabsf(1.0f - scale)*scale*q12 + scale*fabsf(1.0f - scale)*q21 + scale*scale*q22;
}


__global__ void resize(uchar* d_input, int height, int width, uchar* d_output, float scale)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    for (uint i = row; i < height; i += blockDim.y*gridDim.y)
        for (uint j = col; j < width; j += blockDim.x*gridDim.x)
        {
            if (threadIdx.y + 1 < blockDim.y)
            {
                int r = i*scale, c = j*scale;
                float q11 = d_input[r*width + c];
                float q12 = d_input[(r + 1)*width + c];
                float q21 = d_input[r*width + (c + 1)];
                float q22 = d_input[(r + 1)*width + (c + 1)];

                // Bilinear Interpolation
                uchar p = bilinear(q11, q12, q21, q22, scale);
                d_output[i*width + j] = p;
            }
        }
}


void cudaResize(const cv::Mat & input, cv::Mat & output, float scale)
{
    int newRow = int(input.rows * scale);
    int newCol = int(input.cols * scale);
    output = cv::Mat(cv::Size(newCol, newRow), CV_8U, cv::Scalar(0));
    scale = 1.0f / scale;

    // define block size and thread size
    dim3 block_size(THREAD_MULTIPLE, 6);
    dim3 grid_size(output.cols / (4 * block_size.x), output.rows / (4 * block_size.y)); // I divide the image into 16 grid to increase ILP level.

    cudaStream_t stream; cudaStreamCreate(&stream);

    uchar *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows);
    cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*sizeof(uchar)*input.cols, cudaMemcpyHostToDevice, stream);
    cudaMalloc(&d_output, sizeof(uchar)*output.cols*output.rows);

    resize <<<grid_size, block_size, 0, stream >>>(d_input, output.rows, output.cols, d_output, scale);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data, d_output, sizeof(uchar)*output.cols* output.rows, cudaMemcpyDeviceToHost);

    // resource releasing
    cudaStreamDestroy(stream);
    cudaFree(d_input); cudaFree(d_output);
}