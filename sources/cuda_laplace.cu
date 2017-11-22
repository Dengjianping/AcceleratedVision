#include "..\include\cuda_laplace.h"


#define K_SIZE 3
#define RADIUS 1


/*
for instance, sobel operator cane be represented like this sobelKernelXC[3][3] = [1; 2; 1]*[-1, 0, 1], O(m*m) to O(m+m)
so we can convolve image using row-wised and column-wised kernel;
of cause there're lots of filters can be separed or not.
follow this link, list some common filters, https://dsp.stackexchange.com/questions/7586/common-use-cases-for-2d-nonseparable-convolution-filters
*/
__constant__ float laplace_mask[K_SIZE][K_SIZE] = { { 0.0f,1.0f,0.0f },{ 1.0f,-4.0f,1.0f },{ 0.0f,1.0f,0.0f } };
__constant__ float laplace_mask_exetended[K_SIZE][K_SIZE] = { { -1.0f,-1.0f,-1.0f },{ -1.0f,8.0f,-1.0f },{ -1.0f,-1.0f,-1.0f } };


template<LAPLACE_TYPE lp_type>
__global__ void laplace(uchar *d_input, int height, int width, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = 4 * blockDim.x*blockIdx.x + threadIdx.x; // each thread handle 4 pixels

    auto pixel = [](const float & p)
    {
        uchar q = (uchar)(p > 255.0f ? 255 : p);
        return q;
    };

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

        float sum = 0.0f, sum_32 = 0.0f, sum_64 = 0.0f, sum_96 = 0.0f;
        for (int i = -RADIUS; i <= RADIUS; i++)
            for (int j = -RADIUS; j <= RADIUS; j++)
            {
                if (lp_type)
                {
                    sum = fmaf(laplace_mask_exetended[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j], sum);
                    sum_32 = fmaf(laplace_mask_exetended[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 32], sum_32);
                    sum_64 = fmaf(laplace_mask_exetended[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 64], sum_64);
                    sum_96 = fmaf(laplace_mask_exetended[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 96], sum_96);
                }
                else
                {
                    sum = fmaf(laplace_mask[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j], sum);
                    sum_32 = fmaf(laplace_mask[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 32], sum_32);
                    sum_64 = fmaf(laplace_mask[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 64], sum_64);
                    sum_96 = fmaf(laplace_mask[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 96], sum_96);
                }
            }

        d_output[index] = pixel(sum);
        d_output[index + 32] = pixel(sum_32);
        d_output[index + 64] = pixel(sum_64);
        d_output[index + 96] = pixel(sum_96);
    }
}


void cudaLaplace(const cv::Mat & input, cv::Mat & output, LAPLACE_TYPE lp_type)
{
    if (input.channels() != 1)return;
    output = cv::Mat(input.size(), input.type(), cv::Scalar(0));

    cudaStream_t stream; CUDA_CALL(cudaStreamCreate(&stream));

    uchar *d_input, *d_output;
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.rows*input.cols));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.rows*input.cols, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&d_output, sizeof(uchar)*input.rows*input.cols));

    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 6);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (4 * block_size.x), input.rows / (4 * block_size.y));

    switch (lp_type)
    {
    case NORMAL:
        laplace<NORMAL> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output); break;
    case EXTENDED:
        laplace<EXTENDED> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output); break;
    default:
        break;
    }
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(output.data, d_output, sizeof(uchar)*input.rows*input.cols, cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output)); CUDA_CALL(cudaStreamDestroy(stream));
}