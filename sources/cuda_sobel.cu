#include "..\include\cuda_sobel.h"


#define K_SIZE 3
#define RADIUS 1
//#define FAST_MATH


/*
for instance, sobel operator cane be represented like this sobelKernelXC[3][3] = [1; 2; 1]*[-1, 0, 1], O(m*m) to O(m+m)
so we can convolve image using row-wised and column-wised kernel;
of cause there're lots of filters can be separed or not.
follow this link, list some common filters, https://dsp.stackexchange.com/questions/7586/common-use-cases-for-2d-nonseparable-convolution-filters
*/
__constant__ float sobelKernelXC[K_SIZE][K_SIZE] = { { -1.0f,0.0f,1.0f },{ -2.0f,0.0f,2.0f },{ -1.0f,0.0f,1.0f } };
__constant__ float sobelKernelYC[K_SIZE][K_SIZE] = { { -1.0f,-2.0f,-1.0f },{ 0.0f,0.0f,0.0f },{ 1.0f,2.0f,1.0f } };


__global__ void sobel(uchar *d_input, int height, int width, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = 4 * blockDim.x*blockIdx.x + threadIdx.x; // each thread handle 4 pixels

    static __shared__ float smem[6 + 2 * RADIUS][32 * 4 + 2 * RADIUS];

    for (int i = row; i < height; i += blockDim.y*gridDim.y)
    {
        int index = i*width + col;
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

        // load upper-left corner
        if (threadIdx.x < RADIUS && threadIdx.y < RADIUS)
            smem[threadIdx.y][threadIdx.x] = d_input[(i - RADIUS)*width + (col - RADIUS)];

        // load upper-right
        if (threadIdx.x + RADIUS >= 32 && threadIdx.y < RADIUS)
            smem[threadIdx.y][threadIdx.x + 2 * RADIUS + 96] = d_input[(i - RADIUS)*width + (col + RADIUS + 96)];

        // bottom-left
        if (threadIdx.y + RADIUS >= 6 && threadIdx.x < RADIUS)
            smem[threadIdx.y + 2 * RADIUS][threadIdx.x] = d_input[(i + RADIUS)*width + (col - RADIUS)];

        // bottom-right
        if (threadIdx.y + RADIUS >= 6 && threadIdx.x + RADIUS >= 32)
            smem[threadIdx.y + 2 * RADIUS][threadIdx.x + 2 * RADIUS + 96] = d_input[(i + RADIUS)*width + (col + RADIUS + 96)];

        __syncthreads();

        float sumx = 0.0f, sumy = 0.0f, sumx_32 = 0.0f, sumy_32 = 0.0f, sumx_64 = 0.0f, sumy_64 = 0.0f, sumx_96 = 0.0f, sumy_96 = 0.0f;
        for (int i = -RADIUS; i <= RADIUS; i++)
            for (int j = -RADIUS; j <= RADIUS; j++)
            {
                /* sumx += sobelKernelXC[1 + i][1 + j] * smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j];
                sumy += sobelKernelYC[1 + i][1 + j] * smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j];
                sumx_32 += sobelKernelXC[1 + i][1 + j] * smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 32];
                sumy_32 += sobelKernelYC[1 + i][1 + j] * smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 32];
                sumx_64 += sobelKernelXC[1 + i][1 + j] * smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 64];
                sumy_64 += sobelKernelYC[1 + i][1 + j] * smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 64];
                sumx_96 += sobelKernelXC[1 + i][1 + j] * smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 96];
                sumy_96 += sobelKernelYC[1 + i][1 + j] * smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 96];*/

                sumx = fmaf(sobelKernelXC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j], sumx);
                sumy = fmaf(sobelKernelYC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j], sumy);
                sumx_32 = fmaf(sobelKernelXC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 32], sumx_32);
                sumy_32 = fmaf(sobelKernelYC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 32], sumy_32);
                sumx_64 = fmaf(sobelKernelXC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 64], sumx_64);
                sumy_64 = fmaf(sobelKernelYC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 64], sumy_64);
                sumx_96 = fmaf(sobelKernelXC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 96], sumx_96);
                sumy_96 = fmaf(sobelKernelYC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 96], sumy_96);
            }

        auto gradient = [](float *a, float *b)
        {
        #ifndef FAST_MATH
            float x = powf(*a, 2.0f);
            float y = powf(*b, 2.0f);
            return sqrtf(x + y);
        #else
            float x = __powf(*a, 2.0f);
            float y = __powf(*b, 2.0f);
            return sqrtf(x + y);
        #endif
        };

        d_output[index] = gradient(&sumx, &sumy);
        d_output[index + 32] = gradient(&sumx_32, &sumy_32);
        d_output[index + 64] = gradient(&sumx_64, &sumy_64);
        d_output[index + 96] = gradient(&sumx_96, &sumy_96);
    }
}


void cudaSobel(const cv::Mat & input, cv::Mat & output)
{
    if (input.channels() != 1)return;
    output = cv::Mat(input.size(), input.type(), cv::Scalar(0));

    cudaStream_t stream; CUDA_CALL(cudaStreamCreate(&stream));

    uchar *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(uchar)*input.rows*input.cols);
    cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.rows*input.cols, cudaMemcpyHostToDevice, stream);
    cudaMalloc(&d_output, sizeof(uchar)*input.rows*input.cols);

    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 6);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (4 * block_size.x), input.rows / (4 * block_size.y));

    sobel <<<grid_size, block_size, 0, stream>>>(d_input, input.rows, input.cols, d_output);
    cudaDeviceSynchronize();

    cudaMemcpyAsync(output.data, d_output, sizeof(uchar)*input.rows*input.cols, cudaMemcpyDeviceToHost, stream);
    cudaFree(d_input); cudaFree(d_output); cudaStreamDestroy(stream);
}