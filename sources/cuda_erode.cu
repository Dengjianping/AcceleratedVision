#include "../include/cuda_erode.h"


__constant__ int erode_mask[21]; // only support a mask size 21 * 21


template<int RADIUS>
__global__ void erode(uchar *d_input, int height, int width, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = 4 * blockDim.x*blockIdx.x + threadIdx.x; // each thread handle 4 pixels

    static __shared__ int smem[6 + 2 * RADIUS][32 * 4 + 2 * RADIUS];
    for (int i = row; i < height; i += blockDim.y*gridDim.y)
    {
        int index = i*width + col;

        // separatable row computation
        smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS] = d_input[index];
        smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS + 32] = d_input[index + 32];
        smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS + 64] = d_input[index + 64];
        smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS + 96] = d_input[index + 96];

        // up and bottom row
        if (threadIdx.y < RADIUS)
        {
            int global_index = (i - RADIUS)*width + col;
            smem[threadIdx.y][threadIdx.x + RADIUS] = d_input[global_index];
            smem[threadIdx.y][threadIdx.x + RADIUS + 32] = d_input[global_index + 32];
            smem[threadIdx.y][threadIdx.x + RADIUS + 64] = d_input[global_index + 64];
            smem[threadIdx.y][threadIdx.x + RADIUS + 96] = d_input[global_index + 96];
        }
        if (threadIdx.y + RADIUS >= 6)
        {
            int global_index = (i + RADIUS)*width + col;
            smem[threadIdx.y + 2 * RADIUS][threadIdx.x + RADIUS] = d_input[global_index];
            smem[threadIdx.y + 2 * RADIUS][threadIdx.x + RADIUS + 32] = d_input[global_index + 32];
            smem[threadIdx.y + 2 * RADIUS][threadIdx.x + RADIUS + 64] = d_input[global_index + 64];
            smem[threadIdx.y + 2 * RADIUS][threadIdx.x + RADIUS + 96] = d_input[global_index + 96];
        }

        // left and right column
        if (threadIdx.x < RADIUS)
            smem[threadIdx.y + RADIUS][threadIdx.x] = d_input[i*width + (col - RADIUS)];
        if (threadIdx.x + RADIUS >= 32)
            smem[threadIdx.y + RADIUS][threadIdx.x + 2 * RADIUS + 96] = d_input[i*width + col + RADIUS + 96];
        __syncthreads();

        int sum = 255, sum_32 = 255, sum_64 = 255, sum_96 = 255;
        #pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++)
        {
            sum = (erode_mask[RADIUS + j] * smem[threadIdx.y + RADIUS - j][threadIdx.x + RADIUS]) & sum;
            sum_32 = (erode_mask[RADIUS + j] * smem[threadIdx.y + RADIUS - j][threadIdx.x + RADIUS + 32]) & sum_32;
            sum_64 = (erode_mask[RADIUS + j] * smem[threadIdx.y + RADIUS - j][threadIdx.x + RADIUS + 64]) & sum_64;
            sum_96 = (erode_mask[RADIUS + j] * smem[threadIdx.y + RADIUS - j][threadIdx.x + RADIUS + 96]) & sum_96;
        }
        smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS] = sum;
        smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS + 32] = sum_32;
        smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS + 64] = sum_64;
        smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS + 96] = sum_96;
        __syncthreads();

        sum = 255, sum_32 = 255, sum_64 = 255, sum_96 = 255;
        #pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++)
        {
            sum = (erode_mask[RADIUS + j] * smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS - j]) & sum;
            sum_32 = (erode_mask[RADIUS + j] * smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS - j + 32]) & sum_32;
            sum_64 = (erode_mask[RADIUS + j] * smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS - j + 64]) & sum_64;
            sum_96 = (erode_mask[RADIUS + j] * smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS - j + 96]) & sum_96;
        }

        d_output[index] = sum;
        d_output[index + 32] = sum_32;
        d_output[index + 64] = sum_64;
        d_output[index + 96] = sum_96;
    }
}

void cudaErode(const cv::Mat & input, int kernel_size, int eroded_times, cv::Mat & output)
{
    if (input.channels() != 1)return;
    output = cv::Mat(input.size(), input.type(), cv::Scalar(0));

    int diameter = 2 * kernel_size + 1;
    int *host_erode_mask = new int[diameter];
    for (int i = 0; i < diameter; i++)host_erode_mask[i] = 1;
    CUDA_CALL(cudaMemcpyToSymbol(erode_mask, host_erode_mask, sizeof(int)*diameter, 0, cudaMemcpyHostToDevice));

    cudaStream_t stream; CUDA_CALL(cudaStreamCreate(&stream));

    uchar *d_input, *d_output;
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.rows*input.cols));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.rows*input.cols, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&d_output, sizeof(uchar)*input.rows*input.cols));

    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 6);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (4 * block_size.x), input.rows / (4 * block_size.y));

    switch (kernel_size)
    {
    case 1: erode<1> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output); break;
    case 2: erode<2> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output); break;
    case 3: erode<3> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output); break;
    case 4: erode<4> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output); break;
    case 5: erode<5> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output); break;
    case 6: erode<6> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output); break;
    case 7: erode<7> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output); break;
    case 8: erode<8> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output); break;
    case 9: erode<9> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output); break;
    case 10: erode<10> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output); break;
    default: break;
    }

    /*for (int i = 0; i < eroded_times; i++)
    {
        erode<1> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output);
        cudaMemset(d_input, 0, sizeof(uchar)*output.rows*output.cols);
        cudaMemcpyAsync(d_input, d_output, sizeof(uchar)*input.rows*input.cols, cudaMemcpyDeviceToDevice, stream);
        if (i!=eroded_times-1)
            cudaMemset(d_output, 0, sizeof(uchar)*output.rows*output.cols);
    }*/
    //erode<1> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(output.data, d_output, sizeof(uchar)*input.rows*input.cols, cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaFree(d_input));  CUDA_CALL(cudaFree(d_output)); CUDA_CALL(cudaStreamDestroy(stream));
    delete[] host_erode_mask;
}