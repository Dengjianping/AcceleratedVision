#include "../include/cuda_histogram.h"

#define BINS 256


__global__ void histogram(uchar *d_input, int height, int width, uint *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    for (int i = row; i < height; i += blockDim.y*gridDim.y)
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            // atomic function at device with cc 2.1 has a really performance, cc > 3.5 has better performance
            // see this link, https://devblogs.nvidia.com/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
            atomicAdd(&d_output[d_input[i*width + j]], 1);
        }
}


void cudaHistogram(const cv::Mat & input, uint *hist)
{
    hist = new uint[BINS];
    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 8);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (4 * block_size.x), input.rows / (4 * block_size.y));

    uchar *d_input; uint *d_output;
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&d_output, sizeof(uint)*BINS));
    CUDA_CALL(cudaMemset(d_output, 0, sizeof(uint)*BINS));

    // calling kernel
    histogram <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(hist, d_output, sizeof(uint)*BINS, cudaMemcpyDeviceToHost));

    // resources releasing
    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
}