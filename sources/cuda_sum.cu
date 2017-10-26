#include "..\include\cuda_sum.h"


template<int N>
__global__ void reduction(uchar *d_input, int height, int width, int *aux)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = 2 * blockDim.x*blockIdx.x + threadIdx.x;
    __shared__ int smem[17 * 33]; // add one more column to eliminate bank conflict

    for (int i = row; i < height; i += gridDim.y*blockDim.y)
    {
        int r = (threadIdx.x >> 5), c = (threadIdx.x & 0x1f); // i % n equal to i&(n-1) if n is a power of 2
        int index = i*width + col;
        smem[r * 33 + c] = (int)d_input[index];
        smem[(r + 8) * 33 + c] = (int)d_input[index + 256];
        __syncthreads();

        int offset = 1;

        // __shfl_down; // cc 2.1 does not support shuffle function, at least 3.0

        //#pragma unroll
        for (int k = N >> 1; k > 0; k >>= 1)
        {
            //__syncthreads(); // if place a sync here, the result will be wrong, always less the right one, figuring it out.
            if (threadIdx.x < k)
            {
                int ai = offset*(2 * threadIdx.x + 1) - 1;
                int bi = offset*(2 * threadIdx.x + 2) - 1;
                int n = (ai & 0x1f) + (ai >> 5) * 33; // use shift to replace /, & to replace %
                int m = (bi & 0x1f) + (bi >> 5) * 33;
                smem[m] += smem[n];
            }
            __syncthreads();
            offset <<= 1;
        }
        if (threadIdx.x == 0)aux[i*gridDim.x + blockIdx.x] = smem[526]; // store each block sum
    }
}


// add all fragment result
__global__ void sumOfPixels(int N, int *aux, int *d_sum)
{
    for (int i = 0; i < N; i += 256)
    {
        atomicAdd(d_sum, aux[i + threadIdx.x]);
    }
}


//extern "C"
void cudaSum(const cv::Mat & input, int & sum_of_pixels)
{
    dim3 block_size(256, 1);
    dim3 grid_size(input.cols / (2 * block_size.x), input.rows / (8 * block_size.y));

    cudaStream_t stream; cudaStreamCreate(&stream);
    uchar *d_input;
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows, cudaMemcpyHostToDevice, stream));

    int *aux; // store each block of sum
    CUDA_CALL(cudaMalloc(&aux, sizeof(int)*grid_size.x*grid_size.y * 8));

    int *d_sum; // sum of all pixels
    CUDA_CALL(cudaMalloc(&d_sum, sizeof(int))); CUDA_CALL(cudaMemset(d_sum, 0, sizeof(int)));

    reduction<512><<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, aux);
    sumOfPixels<<<1, block_size, 0, stream>>> (grid_size.x*grid_size.y * 8, aux, d_sum);
    CUDA_CALL(cudaDeviceSynchronize()); //CUDA_CALL(cudaStreamSynchronize(stream));

    CUDA_CALL(cudaMemcpy(&sum_of_pixels, d_sum, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(aux)); CUDA_CALL(cudaFree(d_sum));
}


//extern "C"
void cudaMeanValue(const cv::Mat & input, float & mean)
{
    int sum = 0;
    cudaSum(input, sum);
    mean = (float)sum / (float)(input.rows*input.cols);
}