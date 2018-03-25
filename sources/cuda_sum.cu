#include "..\include\cuda_sum.h"


template<uint blockSize>
__global__ void reduction(uchar *d_input, int *d_output, uint n)
{
    __shared__ int smem[256];

    uint i = blockIdx.x*(blockSize * 2) + threadIdx.x; // each block handle 512 pxels
    uint gridSize = blockDim.x * 2 * gridDim.x;

    smem[threadIdx.x] = 0;
    while (i < n)
    {
        smem[threadIdx.x] += d_input[i] + d_input[i + blockSize];
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 256)
    {
        if (threadIdx.x < 128)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 128];
        }
        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (threadIdx.x < 64) { smem[threadIdx.x] += smem[threadIdx.x + 64]; } __syncthreads();
    }

    if (threadIdx.x < 32)
    {
        if (blockSize >= 64)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 32];
        };
        __syncthreads();

        if (blockSize >= 32) smem[threadIdx.x] += smem[threadIdx.x + 16];
        __syncthreads();

        if (blockSize >= 16) smem[threadIdx.x] += smem[threadIdx.x + 8];
        __syncthreads();

        if (blockSize >= 8) smem[threadIdx.x] += smem[threadIdx.x + 4];
        __syncthreads();

        if (blockSize >= 4) smem[threadIdx.x] += smem[threadIdx.x + 2];
        __syncthreads();

        if (blockSize >= 2) smem[threadIdx.x] += smem[threadIdx.x + 1];
        __syncthreads();
    }

    if (threadIdx.x == 0) d_output[blockIdx.x] = smem[0];
}


int cudaSum(const cv::Mat & input)
{
    dim3 block_size(256, 1);
    dim3 grid_size(input.cols*input.rows / (32 * block_size.x), 1);

    uchar *d_input;
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows, cudaMemcpyHostToDevice));

    int *d_output;
    CUDA_CALL(cudaMalloc(&d_output, sizeof(int)*grid_size.x));

    int *h_output = new int[grid_size.x];

    reduction<256> << <grid_size, block_size >> > (d_input, d_output, input.rows*input.cols);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(h_output, d_output, sizeof(int)*grid_size.x, cudaMemcpyDeviceToHost));

    int sum = 0;
    for (int i = 0; i < grid_size.x; i++) sum += h_output[i];

    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
    delete[] h_output;

    return sum;
}


float cudaMean(const cv::Mat & input)
{
    int sum = cudaSum(input);;
    return sum / (float)(input.rows*input.cols);
}