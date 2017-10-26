#include "../include/cuda_transform.h"
#define LOOP_UNROLL


// Bilinear Interpolation
__device__ float bilinear(float q11, float q12, float q21, float q22, float scale)
{
    return (1.0f - scale)*(1.0f - scale)*q11 + (1.0f - scale)*scale*q12 + scale*(1.0f - scale)*q21 + scale*scale*q22;
}


//__global__ void __launch_bounds__(MAX_BLOCK_SIZE, MIN_BLOCKS_PER_SM) resize(uchar* d_input, size_t in_pitch, int height, int width, uchar* d_output, size_t out_pitch, float scale)
__global__ void resize(uchar* d_input, size_t in_pitch, int height, int width, uchar* d_output, size_t out_pitch, float scale)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    for (uint i = row; i < height; i += blockDim.y*gridDim.y)
        //#pragma unroll
        for (uint j = col; j < width; j += blockDim.x*gridDim.x)
        {
#ifdef LOOP_UNROLL
            if (threadIdx.y + 1 < blockDim.y)
            {
                int r = i*scale, c = j*scale;
                uchar *q11 = (uchar*)((char*)d_input + r*in_pitch) + c;
                uchar *q12 = (uchar*)((char*)d_input + (r + 1)*in_pitch) + c;
                uchar *q21 = (uchar*)((char*)d_input + r*in_pitch) + c + 1;
                uchar *q22 = (uchar*)((char*)d_input + (r + 1)*in_pitch) + c + 1;

                // Bilinear Interpolation
                float p = bilinear(*q11, *q12, *q21, *q22, scale);
                uchar *outputPixel = (uchar*)((char*)d_output + i*out_pitch) + j;
                *outputPixel = (uchar)p;

                r = r + 1;
                q11 = (uchar*)((char*)d_input + r*in_pitch) + c;
                q12 = (uchar*)((char*)d_input + (r + 1)*in_pitch) + c;
                q21 = (uchar*)((char*)d_input + r*in_pitch) + c + 1;
                q22 = (uchar*)((char*)d_input + (r + 1)*in_pitch) + c + 1;

                p = bilinear(*q11, *q12, *q21, *q22, scale);
                outputPixel = (uchar*)((char*)d_output + (i + 1)*out_pitch) + j;
                *outputPixel = (uchar)p;
            }
#else
#pragma unroll
            for (uint k = 0; k < 2; k++)
            {
                if (threadIdx.y + 1 < blockDim.y)
                {

                    int r = i*scale + k, c = j*scale;
                    uchar *q11 = (uchar*)((char*)d_input + r*in_pitch) + c;
                    uchar *q12 = (uchar*)((char*)d_input + (r + 1)*in_pitch) + c;
                    uchar *q21 = (uchar*)((char*)d_input + r*in_pitch) + c + 1;
                    uchar *q22 = (uchar*)((char*)d_input + (r + 1)*in_pitch) + c + 1;

                    uchar *outputPixel = (uchar*)((char*)d_output + (i + k)*out_pitch) + j;

                    float p = bilinear(*q11, *q12, *q21, *q22, scale);
                    *outputPixel = (uchar)p;
                }
            }
#endif // 
        }
}


__global__ void tranpose(uchar *d_input, int height, int width, uchar *d_output)
{
    int row = 32 * blockIdx.y + threadIdx.y;
    int col = 32 * blockIdx.x + threadIdx.x;

    __shared__ int smem[32][32 + 1];
    if (row < height&&col < width)
    {
        //#pragma unroll
        for (size_t i = 0; i < 32; i += 8)
        {
            smem[threadIdx.y + i][threadIdx.x] = d_input[(row + i)*width + col];
        }
        __syncthreads();

        row = blockIdx.x*32 + threadIdx.y;
        col = blockIdx.y*32 + threadIdx.x;

        //#pragma unroll
        for (size_t i = 0; i < 32; i += 8)
        {
            d_output[(height - row - i)*width + width - col] = smem[threadIdx.x][threadIdx.y + i];
            //d_output[(width - col - i)*width + height - row] = smem[threadIdx.x][threadIdx.y + i];
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

    size_t in_pitch, out_pitch;
    uchar *d_input, *d_output;
    cudaMallocPitch(&d_input, &in_pitch, sizeof(uchar)*input.cols, input.rows);
    cudaMemcpy2DAsync(d_input, in_pitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, stream);
    cudaMallocPitch(&d_output, &out_pitch, sizeof(uchar)*output.cols, output.rows);

    resize <<<grid_size, block_size, 0, stream >>>(d_input, in_pitch, output.rows, output.cols, d_output, out_pitch, scale);
    cudaDeviceSynchronize();

    cudaMemcpy2D(output.data, sizeof(uchar)*output.cols, d_output, out_pitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyDeviceToHost);

    // resource releasing
    cudaStreamDestroy(stream);
    cudaFree(d_input); cudaFree(d_output);
}