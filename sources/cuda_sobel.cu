#include "..\include\cuda_sobel.h"


enum GRADIENT_DIRECTION { X, Y };


/*
for instance, sobel operator cane be represented like this sobelKernelXC[3][3] = [1; 2; 1]*[-1, 0, 1], O(m*m) to O(m+m)
so we can convolve image using row-wised and column-wised kernel;
of cause there're lots of filters can be separed or not.
follow this link, list some common filters, https://dsp.stackexchange.com/questions/7586/common-use-cases-for-2d-nonseparable-convolution-filters
*/
__constant__ float sobelKernelXC[3][3] = { { -1.0f,0.0f,1.0f },{ -2.0f,0.0f,2.0f },{ -1.0f,0.0f,1.0f } };
__constant__ float sobelKernelYC[3][3] = { { -1.0f,-2.0f,-1.0f },{ 0.0f,0.0f,0.0f },{ 1.0f,2.0f,1.0f } };


template<GRADIENT_DIRECTION direction, int RADIUS>
__global__ void sobel_gradient(uchar *d_input, int height, int width, uchar *d_output)
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

        float sum = 0.0f, sum_32 = 0.0f, sum_64 = 0.0f, sum_96 = 0.0f;
        for (int i = -RADIUS; i <= RADIUS; i++)
            for (int j = -RADIUS; j <= RADIUS; j++)
            {
                if (direction)
                {
                    // y direction
                    sum = fmaf(sobelKernelYC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j], sum);
                    sum_32 = fmaf(sobelKernelYC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 32], sum_32);
                    sum_64 = fmaf(sobelKernelYC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 64], sum_64);
                    sum_96 = fmaf(sobelKernelYC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 96], sum_96);
                }
                else
                {
                    // x direction
                    sum = fmaf(sobelKernelXC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j], sum);
                    sum_32 = fmaf(sobelKernelXC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 32], sum_32);
                    sum_64 = fmaf(sobelKernelXC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 64], sum_64);
                    sum_96 = fmaf(sobelKernelXC[RADIUS + i][RADIUS + j], smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 96], sum_96);
                }
            }

        d_output[index] = sum;
        d_output[index + 32] = sum_32;
        d_output[index + 64] = sum_64;
        d_output[index + 96] = sum_96;
    }
}


__global__ void sobel_get_amplitude(uchar *gradient_x, uchar *gradient_y, int height, int width, uchar *amplitude)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    auto saturate_uchar = [](const float & p)
    {
        uchar q = (uchar)(p > 255.0f ? 255 : p);
        return q;
    };

    auto calc_amplitude = [](const uchar4 *x, const uchar4 *y)
    {
        float amp_x = __powf(x->x, 2.0f) + __powf(y->x, 2.0f);
        float amp_y = __powf(x->y, 2.0f) + __powf(y->y, 2.0f);
        float amp_z = __powf(x->z, 2.0f) + __powf(y->z, 2.0f);
        float amp_w = __powf(x->w, 2.0f) + __powf(y->w, 2.0f);
        amp_x = sqrtf(amp_x);
        amp_y = sqrtf(amp_y);
        amp_z = sqrtf(amp_z);
        amp_w = sqrtf(amp_w);
        uchar u_x = (uchar)(amp_x > 255.0f ? 255 : amp_x);
        uchar u_y = (uchar)(amp_y > 255.0f ? 255 : amp_y);
        uchar u_z = (uchar)(amp_z > 255.0f ? 255 : amp_z);
        uchar u_w = (uchar)(amp_w > 255.0f ? 255 : amp_w);
        return make_uchar4(u_x, u_y, u_z, u_w);
    };

    for (int i = row; i < height / 4; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            int index = i*width + j;
            auto x = reinterpret_cast<uchar4*>(gradient_x)[index];
            auto y = reinterpret_cast<uchar4*>(gradient_y)[index];

            reinterpret_cast<uchar4*>(amplitude)[index] = calc_amplitude(&x, &y);
        }
}


void cudaSobel(const cv::Mat & input, cv::Mat & output)
{
    if (input.channels() != 1)return;
    output = cv::Mat(input.size(), input.type(), cv::Scalar(0));

    cudaStream_t stream_x, stream_y; 
    CUDA_CALL(cudaStreamCreate(&stream_x)); CUDA_CALL(cudaStreamCreate(&stream_y));

    uchar *d_input, *d_output;
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.rows*input.cols));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.rows*input.cols, cudaMemcpyHostToDevice, stream_x));
    CUDA_CALL(cudaMalloc(&d_output, sizeof(uchar)*input.rows*input.cols));

    // gradient matrix
    uchar *gradient_x, *gradient_y;
    CUDA_CALL(cudaMalloc(&gradient_x, sizeof(uchar)*input.rows*input.cols));
    CUDA_CALL(cudaMalloc(&gradient_y, sizeof(uchar)*input.rows*input.cols));

    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 6);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (4 * block_size.x), input.rows / (4 * block_size.y));

    sobel_gradient<X, 1> <<<grid_size, block_size, 0, stream_x>>> (d_input, input.rows, input.cols, gradient_x);
    sobel_gradient<Y, 1> <<<grid_size, block_size, 0, stream_y>>> (d_input, input.rows, input.cols, gradient_y);
    CUDA_CALL(cudaStreamSynchronize(stream_y));
    sobel_get_amplitude <<<grid_size, block_size, 0, stream_x>>>(gradient_x, gradient_y, input.rows, input.cols, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(output.data, d_output, sizeof(uchar)*input.rows*input.cols, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output)); CUDA_CALL(cudaFree(gradient_x)); CUDA_CALL(cudaFree(gradient_y));
    CUDA_CALL(cudaStreamDestroy(stream_x)); CUDA_CALL(cudaStreamDestroy(stream_y));
}