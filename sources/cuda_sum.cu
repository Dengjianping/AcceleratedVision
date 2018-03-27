#include "..\include\cuda_sum.h"


template<uint blockSize>
__global__ void reduction(uchar *d_input, uint *d_output, uint n)
{
    __shared__ uint smem[256];

    uint i = blockIdx.x*(blockSize * 2) + threadIdx.x;
    uint gridSize = 2 * blockDim.x * gridDim.x;

    smem[threadIdx.x] = 0;
    while (i < n / 4)
    {
        // each block handle 256 * 4 *2 pxels, 256 threads in a block, which will read 256 * 4 pixels
        // and while reads another 1024 pixels, this will improve 3x performance than just handle 256 * 1 * 2
        uchar4 p0 = reinterpret_cast<uchar4*>(d_input)[i];
        uchar4 p1 = reinterpret_cast<uchar4*>(d_input)[i + blockSize];
        smem[threadIdx.x] += (p0.x + p1.x) + (p0.y + p1.y) + (p0.z + p1.z) + (p0.w + p1.w);
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

    if (threadIdx.x == 0) d_output[blockIdx.x] += smem[0]; // store result to global memory
}


template<uint blockSize>
__global__ void reduction_three_channels(uchar *d_input, uint *d_output, uint n)
{
    static __shared__ uint smem[256 * 3]; // each block handle 256 * 2 * 3 pixels
    static __shared__ uint result[3][256]; // store each channel to this share memory

    uint i = 3 * blockIdx.x*(blockSize * 2) + threadIdx.x;
    uint gridSize = 3 * blockDim.x * 2 * gridDim.x;

    smem[threadIdx.x] = 0; smem[threadIdx.x + 256] = 0; smem[threadIdx.x + 512] = 0;
    result[0][threadIdx.x] = 0; result[1][threadIdx.x] = 0; result[2][threadIdx.x] = 0;
    __syncthreads();
    while (i < 3 * n)
    {
        /*smem[threadIdx.x] += d_input[i] + d_input[i + 3 * blockSize];
        smem[threadIdx.x + 256] += d_input[i + blockSize] + d_input[i + 4 * blockSize];
        smem[threadIdx.x + 512] += d_input[i + 2 * blockSize] + d_input[i + 5 * blockSize];*/
        uchar *p = &d_input[i]; // use this way, saved 3 registers here
        smem[threadIdx.x] += *p + *(p + 3 * blockSize);
        smem[threadIdx.x + blockSize] += *(p + blockSize) + *(p + 4 * blockSize);
        smem[threadIdx.x + 2 * blockSize] += *(p + 2 * blockSize) + *(p + 5 * blockSize);
        __syncthreads(); // mother fxxk, wasted all most half day here. remember to wait all addition done, otherwise the result will be diffrent time to time

        result[0][threadIdx.x] = smem[3 * threadIdx.x];
        result[1][threadIdx.x] = smem[3 * threadIdx.x + 1];
        result[2][threadIdx.x] = smem[3 * threadIdx.x + 2];
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 256)
    {
        if (threadIdx.x < 128)
        {
            result[0][threadIdx.x] += result[0][threadIdx.x + 128];
            result[1][threadIdx.x] += result[1][threadIdx.x + 128];
            result[2][threadIdx.x] += result[2][threadIdx.x + 128];
        }
        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (threadIdx.x < 64) 
        { 
            result[0][threadIdx.x] += result[0][threadIdx.x + 64];
            result[1][threadIdx.x] += result[1][threadIdx.x + 64];
            result[2][threadIdx.x] += result[2][threadIdx.x + 64];
        } 
        __syncthreads();
    }

    if (threadIdx.x < 32)
    {
        if (blockSize >= 64)
        {
            result[0][threadIdx.x] += result[0][threadIdx.x + 32];
            result[1][threadIdx.x] += result[1][threadIdx.x + 32];
            result[2][threadIdx.x] += result[2][threadIdx.x + 32];

        };
        __syncthreads();

        if (blockSize >= 32)
        {
            result[0][threadIdx.x] += result[0][threadIdx.x + 16];
            result[1][threadIdx.x] += result[1][threadIdx.x + 16];
            result[2][threadIdx.x] += result[2][threadIdx.x + 16];
        }
        __syncthreads();

        if (blockSize >= 16)
        {
            result[0][threadIdx.x] += result[0][threadIdx.x + 8];
            result[1][threadIdx.x] += result[1][threadIdx.x + 8];
            result[2][threadIdx.x] += result[2][threadIdx.x + 8];
        }
        __syncthreads();

        if (blockSize >= 8)
        {
            result[0][threadIdx.x] += result[0][threadIdx.x + 4];
            result[1][threadIdx.x] += result[1][threadIdx.x + 4];
            result[2][threadIdx.x] += result[2][threadIdx.x + 4];
        }
        __syncthreads();

        if (blockSize >= 4)
        {
            result[0][threadIdx.x] += result[0][threadIdx.x + 2];
            result[1][threadIdx.x] += result[1][threadIdx.x + 2];
            result[2][threadIdx.x] += result[2][threadIdx.x + 2];
        }
        __syncthreads();

        if (blockSize >= 2)
        {
            result[0][threadIdx.x] += result[0][threadIdx.x + 1];
            result[1][threadIdx.x] += result[1][threadIdx.x + 1];
            result[2][threadIdx.x] += result[2][threadIdx.x + 1];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        d_output[blockIdx.x] += result[0][0];
        d_output[blockIdx.x + gridDim.x] += result[1][0];
        d_output[blockIdx.x + 2 * gridDim.x] += result[2][0];
    }
}


void cudaSum(const cv::Mat & input, int sum_result[3])
{
    dim3 block_size(256, 1);
    dim3 grid_size;
    if (input.cols*input.rows < 512 * 512) grid_size = dim3(input.cols*input.rows / (32 * block_size.x), 1);
    else grid_size = dim3(input.cols*input.rows / (128 * block_size.x), 1); //for image size bigger than 512* 512

    int channels = input.channels();

    uchar *d_input;
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows*channels));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows*channels, cudaMemcpyHostToDevice));

    uint *d_output;
    CUDA_CALL(cudaMalloc(&d_output, sizeof(uint)*grid_size.x*channels));
    CUDA_CALL(cudaMemset(d_output, sizeof(uint)*grid_size.x*channels, 0)); // remenber to set it all value as 0 

    uint *h_output = new uint[grid_size.x*channels];

    switch (channels)
    {
    case 1: reduction<256> <<<grid_size, block_size>>> (d_input, d_output, input.rows*input.cols); break;
    case 3: reduction_three_channels<256> <<<grid_size, block_size>>> (d_input, d_output, input.rows*input.cols); break;
    default: break;
    }
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(h_output, d_output, sizeof(uint)*grid_size.x*channels, cudaMemcpyDeviceToHost));

    sum_result[0] = 0;
    sum_result[1] = 0;
    sum_result[2] = 0;
    for (int i = 0; i < channels; i++)
        for (int j = 0; j < grid_size.x; j++)
        {
            sum_result[i] += h_output[i*grid_size.x + j];
        } // do accumulation on host

    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
    delete[] h_output;
}


void cudaMean(const cv::Mat & input, float mean_result[3])
{
    int sum_result[3] = { 0 };
    cudaSum(input, sum_result);
    float pixels_count = input.rows * input.cols; // turn it to float type
    mean_result[0] = sum_result[0] / pixels_count;
    mean_result[1] = sum_result[1] / pixels_count;
    mean_result[2] = sum_result[2] / pixels_count;
}