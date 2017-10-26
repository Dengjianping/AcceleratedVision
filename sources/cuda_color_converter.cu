#include "..\include\cuda_color_converter.h"

#define RED_WEIGHT 0.2989f
#define GREEN_WEIGHT 0.5870f
#define BLUE_WEIGHT 0.1140f
#define MAX_PIXEL(r, g, b) fmaxf(r, fmaxf(g, b))
#define MIN_PIXEL(r, g, b) fminf(r, fminf(g, b))


__global__ void rgb2gray(uchar *d_input, int height, int width, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x;
    __shared__ uchar smem[6][128 * 3]; // using float will cause bank conflict, and register usage will spill. 

    for (int i = row; i < height; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            int out_index = i*width + j + threadIdx.x;
            int index = 3 * i*width + 3 * j + threadIdx.x;

            reinterpret_cast<uchar4*>(smem[threadIdx.y])[threadIdx.x + 0] = reinterpret_cast<uchar4*>(d_input)[index + 0];
            reinterpret_cast<uchar4*>(smem[threadIdx.y])[threadIdx.x + 32] = reinterpret_cast<uchar4*>(d_input)[index + 32];
            reinterpret_cast<uchar4*>(smem[threadIdx.y])[threadIdx.x + 64] = reinterpret_cast<uchar4*>(d_input)[index + 64];
            __syncthreads();

            uchar4 p0;
            int stride = 12 * threadIdx.x;
            p0.x = (uchar)((float)smem[threadIdx.y][stride + 0] * RED_WEIGHT + (float)smem[threadIdx.y][stride + 1] * GREEN_WEIGHT + \
                (float)smem[threadIdx.y][stride + 2] * BLUE_WEIGHT);
            p0.y = (uchar)((float)smem[threadIdx.y][stride + 3] * RED_WEIGHT + (float)smem[threadIdx.y][stride + 4] * GREEN_WEIGHT + \
                (float)smem[threadIdx.y][stride + 5] * BLUE_WEIGHT);
            p0.z = (uchar)((float)smem[threadIdx.y][stride + 6] * RED_WEIGHT + (float)smem[threadIdx.y][stride + 7] * GREEN_WEIGHT + \
                (float)smem[threadIdx.y][stride + 8] * BLUE_WEIGHT);
            p0.w = (uchar)((float)smem[threadIdx.y][stride + 9] * RED_WEIGHT + (float)smem[threadIdx.y][stride + 10] * GREEN_WEIGHT + \
                (float)smem[threadIdx.y][stride + 11] * BLUE_WEIGHT);

            reinterpret_cast<uchar4*>(d_output)[out_index] = p0;
        }
}


__inline__ __device__ float3 hsv_converter(float3 *rgb)
{
    float h, s, v;
    float r = rgb->x, g = rgb->y, b = rgb->z;
    v = MAX_PIXEL(r, g, b);
    float min = MIN_PIXEL(r, g, b);
    min = v - min;
    float temp = __fdividef(__fmul_rd(255.0f, min), v); // improved from 8.42ms to 5.69ms, 23 registers to 19
    // s = v != 0.0f ? 255.0f*min / v : 0.0f;
    s = v != 0.0f ? temp : 0.0f;

    // float tmp = 60.0f / min;
    float tmp = __fdividef(60.0f, min); // improved from 11.07ms to 8.42ms
    if (v == r)
        h = tmp*(g - b);
    if (v == g)
        h = 120.0f + tmp*(b - r);
    if (v == b)
        h = 240.0f + tmp*(r - b);
    h = h < 0.0f ? 360.0f + h : h;
    //h = h > 180.0f ? h : 180.0f;

    return make_float3(h, s, v);
}


__global__ void rgb2hsv(uchar *d_input, int height, int width, uchar *d_output)
{
    uint row = blockDim.y*blockIdx.y + threadIdx.y;
    __shared__ float smem[6][3 * 32];

    for (uint i = row; i < height; i += blockDim.y*gridDim.y)
    {
        int index = 3 * i*width + 3 * blockDim.x*blockIdx.x + threadIdx.x;
        smem[threadIdx.y][threadIdx.x + 0] = (float)d_input[index];
        smem[threadIdx.y][threadIdx.x + 32] = (float)d_input[index + 32];
        smem[threadIdx.y][threadIdx.x + 64] = (float)d_input[index + 64];
        __syncthreads();

        float3 rgb = make_float3(smem[threadIdx.y][3 * threadIdx.x], smem[threadIdx.y][3 * threadIdx.x + 1], smem[threadIdx.y][3 * threadIdx.x + 2]);
        float3 hsv = hsv_converter(&rgb);
        smem[threadIdx.y][3 * threadIdx.x + 0] = hsv.x;
        smem[threadIdx.y][3 * threadIdx.x + 1] = hsv.y;
        smem[threadIdx.y][3 * threadIdx.x + 2] = hsv.z;
        __syncthreads();

        d_output[index] = smem[threadIdx.y][threadIdx.x + 0];
        d_output[index + 32] = smem[threadIdx.y][threadIdx.x + 32];
        d_output[index + 64] = smem[threadIdx.y][threadIdx.x + 64];
    }
}


void cudaGray(const cv::Mat & input, cv::Mat & output)
{
    if (input.channels() == 1)
    {
        // single channel
        input.copyTo(output);
        return;
    }
    output = cv::Mat(input.size(), CV_8U, cv::Scalar(0));

    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));

    uchar *d_input; uchar *d_output;
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar3)*input.cols*input.rows));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar3)*input.cols*input.rows, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&d_output, sizeof(uchar)*input.cols*input.rows));

    dim3 block_size(THREAD_MULTIPLE, 6);
    // proper grid size will get the best performance, I tried on my laptop, cc 2.1, NVS4200m, devided into 4*4 grids will get the best performance
    // less threads do more things. But caring about register usage.
    dim3 grid_size(input.cols / (4 * block_size.x), input.rows / (4 * block_size.y));
    rgb2gray<<<grid_size, block_size, 0, stream>>> (d_input, input.rows/4, input.cols, d_output);
    CUDA_CALL(cudaDeviceSynchronize()); // block evenrything until the kernel completes execution

    // copy data back
    CUDA_CALL(cudaMemcpy(output.data, d_output, sizeof(uchar)*input.cols*input.rows, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
}


void cudaHSV(const cv::Mat & input, cv::Mat & output)
{
    if (input.channels() == 1)
    {
        std::cout << "This image is a single channel image, please provide a mutil-channle image" << std::endl;
        return;
    }
    output = cv::Mat(input.size(), input.type(), cv::Scalar(0));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    uchar *d_input; uchar *d_output;
    cudaMalloc(&d_input, sizeof(uchar3)*input.cols*input.rows);
    cudaMemcpyAsync(d_input, input.data, sizeof(uchar3)*input.cols*input.rows, cudaMemcpyHostToDevice, stream);
    cudaMalloc(&d_output, sizeof(uchar3)*input.cols*input.rows);

    dim3 block_size(THREAD_MULTIPLE, 6);
    // proper grid size will get the best performance, I tried on my laptop, cc 2.1, NVS4200m, devided into 4*4 grids will get the best performance
    // less threads do more things. But caring about register usage.
    dim3 grid_size(input.cols / (1 * block_size.x), input.rows / (8 * block_size.y));
    rgb2hsv <<<grid_size, block_size, 0, stream >> >(d_input, input.rows, input.cols, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    cudaMemcpy(output.data, d_output, sizeof(uchar3)*input.cols*input.rows, cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream);
    cudaFree(d_input); cudaFree(d_output);
}