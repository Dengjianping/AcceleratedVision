#include "../include/cuda_antisotropy.h"


/*
see reference. https://en.wikipedia.org/wiki/Anisotropic_diffusion
*/

template<int channels>
__global__ void antisotropy(uchar *d_input, int height, int width, float lamdba, int k, int iteration, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = channels * blockDim.x*blockIdx.x + threadIdx.x;

    __shared__ float smem[8][channels * THREAD_MULTIPLE + 2];
    __shared__ float n_smem[8][channels * THREAD_MULTIPLE + 2];
    int index = channels*row*width + col;
    smem[threadIdx.y][threadIdx.x] = d_input[index];
    smem[threadIdx.y][threadIdx.x + 32] = d_input[index + 32];
    smem[threadIdx.y][threadIdx.x + 64] = d_input[index + 64];

    n_smem[threadIdx.y][threadIdx.x] = d_input[index];
    n_smem[threadIdx.y][threadIdx.x + 32] = d_input[index + 32];
    n_smem[threadIdx.y][threadIdx.x + 64] = d_input[index + 64];
    __syncthreads();

    k = 1.0 / __powf(k, 2.0f);
    if (threadIdx.y > 0 && threadIdx.y < 5 && threadIdx.x>0 && threadIdx.x < 31)
    {
        for (int i = 0; i < iteration; i++)
        {
            n_smem[threadIdx.y][channels * threadIdx.x] = smem[threadIdx.y - 1][channels * threadIdx.x] - smem[threadIdx.y][channels * threadIdx.x];
            n_smem[threadIdx.y][channels * threadIdx.x + 1] = smem[threadIdx.y - 1][channels * threadIdx.x + 1] - smem[threadIdx.y][channels * threadIdx.x + 1];
            n_smem[threadIdx.y][channels * threadIdx.x + 2] = smem[threadIdx.y - 1][channels * threadIdx.x + 2] - smem[threadIdx.y][channels * threadIdx.x + 2];
            float nc0 = __expf(-n_smem[threadIdx.y][channels * threadIdx.x]* n_smem[threadIdx.y][channels * threadIdx.x] *k)*n_smem[threadIdx.y][channels * threadIdx.x];
            float nc1 = __expf(-n_smem[threadIdx.y][channels * threadIdx.x + 1]* n_smem[threadIdx.y][channels * threadIdx.x + 1] *k)*n_smem[threadIdx.y][channels * threadIdx.x + 1];
            float nc2 = __expf(-n_smem[threadIdx.y][channels * threadIdx.x + 2]* n_smem[threadIdx.y][channels * threadIdx.x + 2] *k)*n_smem[threadIdx.y][channels * threadIdx.x + 2];

            n_smem[threadIdx.y][channels * threadIdx.x] = smem[threadIdx.y + 1][channels * threadIdx.x] - smem[threadIdx.y][channels * threadIdx.x];
            n_smem[threadIdx.y][channels * threadIdx.x + 1] = smem[threadIdx.y + 1][channels * threadIdx.x + 1] - smem[threadIdx.y][channels * threadIdx.x + 1];
            n_smem[threadIdx.y][channels * threadIdx.x + 2] = smem[threadIdx.y + 1][channels * threadIdx.x + 2] - smem[threadIdx.y][channels * threadIdx.x + 2];
            float sc0 = __expf(-n_smem[threadIdx.y][channels * threadIdx.x] * n_smem[threadIdx.y][channels * threadIdx.x] *k)*n_smem[threadIdx.y][channels * threadIdx.x];
            float sc1 = __expf(-n_smem[threadIdx.y][channels * threadIdx.x + 1] * n_smem[threadIdx.y][channels * threadIdx.x + 1]*k)*n_smem[threadIdx.y][channels * threadIdx.x + 1];
            float sc2 = __expf(-n_smem[threadIdx.y][channels * threadIdx.x + 2] * n_smem[threadIdx.y][channels * threadIdx.x + 2] *k)*n_smem[threadIdx.y][channels * threadIdx.x + 2];

            n_smem[threadIdx.y][channels * threadIdx.x] = smem[threadIdx.y][channels * threadIdx.x - channels] - smem[threadIdx.y][channels * threadIdx.x];
            n_smem[threadIdx.y][channels * threadIdx.x + 1] = smem[threadIdx.y][channels * threadIdx.x - 2] - smem[threadIdx.y][channels * threadIdx.x + 1];
            n_smem[threadIdx.y][channels * threadIdx.x + 2] = smem[threadIdx.y][channels * threadIdx.x - 2] - smem[threadIdx.y][channels * threadIdx.x + 2];
            float ec0 = __expf(-n_smem[threadIdx.y][channels * threadIdx.x] * n_smem[threadIdx.y][channels * threadIdx.x] *k)*n_smem[threadIdx.y][channels * threadIdx.x];
            float ec1 = __expf(-n_smem[threadIdx.y][channels * threadIdx.x + 1] * n_smem[threadIdx.y][channels * threadIdx.x + 1] *k)*n_smem[threadIdx.y][channels * threadIdx.x + 1];
            float ec2 = __expf(-n_smem[threadIdx.y][channels * threadIdx.x + 2] * n_smem[threadIdx.y][channels * threadIdx.x + 2]*k)*n_smem[threadIdx.y][channels * threadIdx.x + 2];

            n_smem[threadIdx.y][channels * threadIdx.x] = smem[threadIdx.y][channels * threadIdx.x + channels] - smem[threadIdx.y][channels * threadIdx.x];
            n_smem[threadIdx.y][channels * threadIdx.x + 1] = smem[threadIdx.y][channels * threadIdx.x + 4] - smem[threadIdx.y][channels * threadIdx.x + 1];
            n_smem[threadIdx.y][channels * threadIdx.x + 2] = smem[threadIdx.y][channels * threadIdx.x + 5] - smem[threadIdx.y][channels * threadIdx.x + 2];
            float wc0 = __expf(-n_smem[threadIdx.y][channels * threadIdx.x] * n_smem[threadIdx.y][channels * threadIdx.x] *k)*n_smem[threadIdx.y][channels * threadIdx.x];
            float wc1 = __expf(-n_smem[threadIdx.y][channels * threadIdx.x + 1] * n_smem[threadIdx.y][channels * threadIdx.x + 1] *k)*n_smem[threadIdx.y][channels * threadIdx.x + 1];
            float wc2 = __expf(-n_smem[threadIdx.y][channels * threadIdx.x + 2] * n_smem[threadIdx.y][channels * threadIdx.x + 2] *k)*n_smem[threadIdx.y][channels * threadIdx.x + 2];

            smem[threadIdx.y][channels * threadIdx.x] = smem[threadIdx.y][channels * threadIdx.x] + lamdba*(nc0 + sc0 + ec0 + wc0);
            smem[threadIdx.y][channels * threadIdx.x + 1] = smem[threadIdx.y][channels * threadIdx.x + 1] + lamdba * (nc1 + sc1 + ec1 + wc1);
            smem[threadIdx.y][channels * threadIdx.x + 2] = smem[threadIdx.y][channels * threadIdx.x + 2] + lamdba * (nc2 + sc2 + ec2 + wc2);
            __syncthreads();
        }
        d_output[index] = smem[threadIdx.y][threadIdx.x];
        d_output[index + 32] = smem[threadIdx.y][threadIdx.x + 32];
        d_output[index + 64] = smem[threadIdx.y][threadIdx.x + 64];
    }
}


void cudaAntisotropy(const cv::Mat & input, float lamdba, int k, int iteration, cv::Mat & output)
{
    int channels = input.channels();

    output = cv::Mat(input.size(), input.type(), cv::Scalar(0));
    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 6);
    dim3 grid_size(input.cols / (1 * block_size.x), input.rows / (1 * block_size.y));

    uchar *d_input, *d_output;
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows*channels));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows*channels, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&d_output, sizeof(uchar)*input.cols*input.rows*channels));

    switch (channels)
    {
    case 1: antisotropy<1><<<grid_size, block_size, 0, stream>>>(d_input, input.rows, input.cols, lamdba, k, iteration, d_output); break;
    case 3: antisotropy<3><<<grid_size, block_size, 0, stream>>>(d_input, input.rows, input.cols, lamdba, k, iteration, d_output); break;
    default: break;
    }
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(output.data, d_output, sizeof(uchar)*output.cols*output.rows*channels, cudaMemcpyDeviceToHost));

    // resources releasing
    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
}