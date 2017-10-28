#include "../include/cuda_arithmetic_logical.h"


__device__ uchar4 operator+(const uchar4 & a, const uchar4 & b)
{
    uchar m = a.x + b.x;
    uchar n = a.y + b.y;
    uchar p = a.z + b.z;
    uchar q = a.w + b.w;
    m = m > 255 ? 255 : m;
    n = m > 255 ? 255 : n;
    p = m > 255 ? 255 : p;
    q = m > 255 ? 255 : q;
    return make_uchar4(m, n, p, q);
}


__device__ uchar4 operator-(const uchar4 & a, const uchar4 & b)
{
    return make_uchar4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}


__global__ void add_one_channel(uchar *d_input1, uchar *d_input2, int height, int width, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    auto add_pixels = [](uchar4 *a, uchar4 *b)
    {
        uchar m = a->x + b->x;
        uchar n = a->y + b->y;
        uchar p = a->z + b->z;
        uchar q = a->w + b->w;
        m = m > 255 ? 255 : m;
        n = m > 255 ? 255 : n;
        p = m > 255 ? 255 : p;
        q = m > 255 ? 255 : q;
        return make_uchar4(m, n, p, q);
    };

    // use blockDim.y*gridDim.y or blockDim.x*gridDim.x to retrieve elements, increasing ILP level, and memory accessing is coalesced.
    // visit this post, https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (uint i = row; i < height / 4; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (uint j = col; j < width; j += blockDim.x*gridDim.x)
        {
            // because use uchar, I just get 25% loading efficiency on global memory, a warp only request only 32 bytes, but ideally 128 bytes.
            // There's a post from CUDA blog, https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
            // as result, it benefit a lot performance improvement.
            uchar4 temp = reinterpret_cast<uchar4*>(d_input1)[i*width + j];
            uchar4 temp1 = reinterpret_cast<uchar4*>(d_input2)[i*width + j];
            reinterpret_cast<uchar4*>(d_output)[i*width + j] = add_pixels(&temp, &temp1);
        }
}


__global__ void add_three_channel(uchar *d_input1, uchar *d_input2, int height, int width, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x;

    auto add_pixels = [](uchar4 *a, uchar4 *b)
    {
        uchar m = a->x + b->x > 255 ? 255 : a->x + b->x;
        uchar n = a->y + b->y > 255 ? 255 : a->y + b->y;
        uchar p = a->z + b->z > 255 ? 255 : a->z + b->z;
        uchar q = a->w + b->w > 255 ? 255 : a->w + b->w;
        return make_uchar4(m, n, p, q);
    };

    for (int i = row; i < height / 4; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            int index = 3 * i*width + 3 * j + threadIdx.x;

            uchar4 p0 = reinterpret_cast<uchar4*>(d_input1)[index + 0];
            uchar4 p1 = reinterpret_cast<uchar4*>(d_input1)[index + 32];
            uchar4 p2 = reinterpret_cast<uchar4*>(d_input1)[index + 64];
            uchar4 q0 = reinterpret_cast<uchar4*>(d_input2)[index + 0];
            uchar4 q1 = reinterpret_cast<uchar4*>(d_input2)[index + 32];
            uchar4 q2 = reinterpret_cast<uchar4*>(d_input2)[index + 64];

            reinterpret_cast<uchar4*>(d_output)[index] = add_pixels(&p0, &q0);
            reinterpret_cast<uchar4*>(d_output)[index + 32] = add_pixels(&p1, &q1);
            reinterpret_cast<uchar4*>(d_output)[index + 64] = add_pixels(&p2, &q2);

        }
}


__global__ void sqrt_pixel(uchar *d_input, int height, int width, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = 1*blockDim.x*blockIdx.x + threadIdx.x;

    auto result = [](uchar4 *a)
    {
        uchar m = (uchar)sqrtf((float)a->x);
        uchar n = (uchar)sqrtf((float)a->y);
        uchar p = (uchar)sqrtf((float)a->z);
        uchar q = (uchar)sqrtf((float)a->w);
        return make_uchar4(m, n, p, q);
    };

    // use blockDim.y*gridDim.y or blockDim.x*gridDim.x to retrieve elements, increasing ILP level, and memory accessing is coalesced.
    // visit this post, https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (uint i = row; i < height / 4; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (uint j = col; j < width; j += blockDim.x*gridDim.x)
        {
            // because use uchar, I just get 25% loading efficiency on global memory, a warp only request only 32 bytes, but ideally 128 bytes.
            // There's a post from CUDA blog, https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
            // as result, it benefit a lot performance improvement.
            uchar4 temp = reinterpret_cast<uchar4*>(d_input)[i*width + j];
            //uchar4 temp1 = reinterpret_cast<uchar4*>(d_input)[i*width + j + 32];
            reinterpret_cast<uchar4*>(d_output)[i*width + j] = result(&temp);
            //reinterpret_cast<uchar4*>(d_output)[i*width + j + 32] = result(&temp1);
        }
}


__global__ void sqrt_three_channel(uchar *d_input, int height, int width, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x;

    auto sqrt_pixels = [](const float4 *a)
    {
        uchar m = (uchar)sqrtf(a->x);
        uchar n = (uchar)sqrtf(a->y);
        uchar p = (uchar)sqrtf(a->z);
        uchar q = (uchar)sqrtf(a->w);
        return make_uchar4(m, n, p, q);
    };

    for (int i = row; i < height / 4; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            int index = 3 * i*width + 3 * j + threadIdx.x;

            uchar4 p0 = reinterpret_cast<uchar4*>(d_input)[index + 0];
            uchar4 p1 = reinterpret_cast<uchar4*>(d_input)[index + 32];
            uchar4 p2 = reinterpret_cast<uchar4*>(d_input)[index + 64];

            float4 q0 = make_float4((float)p0.x, (float)p0.y, (float)p0.z, (float)p0.w);
            float4 q1 = make_float4((float)p1.x, (float)p1.y, (float)p1.z, (float)p1.w);
            float4 q2 = make_float4((float)p2.x, (float)p2.y, (float)p2.z, (float)p2.w);

            reinterpret_cast<uchar4*>(d_output)[index] = sqrt_pixels(&q0);
            reinterpret_cast<uchar4*>(d_output)[index + 32] = sqrt_pixels(&q1);
            reinterpret_cast<uchar4*>(d_output)[index + 64] = sqrt_pixels(&q2);

        }
}


void cudaADD(const cv::Mat & input1, const cv::Mat & input2, cv::Mat & output)
{
    if (input1.channels() != input2.channels())return;
    if (input1.size() != input2.size())return;
    int channel = input1.channels();

    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 6);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input1.cols / (4 * block_size.x), input1.rows / (4 * block_size.y));

    uchar *d_input1, *d_input2, *d_output;
    cudaStream_t stream; cudaStreamCreate(&stream);

    cudaMalloc(&d_input1, channel * sizeof(uchar)*input1.cols*input1.rows);
    cudaMalloc(&d_input2, channel * sizeof(uchar)*input2.cols*input2.rows);
    cudaMalloc(&d_output, channel * sizeof(uchar)*input2.cols*input2.rows);
    cudaMemcpyAsync(d_input1, input1.data, channel * sizeof(uchar)*input1.rows*input1.cols, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_input2, input2.data, channel * sizeof(uchar)*input2.rows*input2.cols, cudaMemcpyHostToDevice, stream);

    switch (channel)
    {
    case 1:
        output = cv::Mat(input1.size(), input1.type(), cv::Scalar(0));
        add_one_channel <<<grid_size, block_size, 0, stream>>> (d_input1, d_input2, input1.rows, input1.cols, d_output);
        cudaDeviceSynchronize();
        break;
    case 3:
        output = cv::Mat(input1.size(), input1.type(), cv::Scalar(0, 0, 0));
        add_three_channel <<<grid_size, block_size, 0, stream>>> (d_input1, d_input2, input1.rows, input1.cols, d_output);
        cudaDeviceSynchronize();
        break;
    default:
        break;
    }

    cudaMemcpy(output.data, d_output, channel * sizeof(uchar)*input1.rows*input2.cols, cudaMemcpyDeviceToHost);

    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_input1)); CUDA_CALL(cudaFree(d_input2)); CUDA_CALL(cudaFree(d_output));
}



void cudaSqrt(const cv::Mat & input, cv::Mat & output)
{
    int channel = input.channels();

    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 4);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (4 * block_size.x), input.rows / (4 * block_size.y));

    uchar *d_input, *d_output;
    cudaStream_t stream; cudaStreamCreate(&stream);

    cudaMalloc(&d_input, channel * sizeof(uchar)*input.cols*input.rows);
    cudaMalloc(&d_output, channel * sizeof(uchar)*input.cols*input.rows);
    cudaMemcpyAsync(d_input, input.data, channel * sizeof(uchar)*input.rows*input.cols, cudaMemcpyHostToDevice, stream);
   
    switch (channel)
    {
    case 1:
        output = cv::Mat(input.size(), input.type(), cv::Scalar(0));
        sqrt_pixel <<<grid_size, block_size, 0, stream >> > (d_input, input.rows, input.cols, d_output);
        cudaDeviceSynchronize();
        break;
    case 3:
        output = cv::Mat(input.size(), input.type(), cv::Scalar(0, 0, 0));
        sqrt_three_channel <<<grid_size, block_size, 0, stream >>> (d_input, input.rows, input.cols, d_output);
        cudaDeviceSynchronize();
        break;
    default:
        break;
    }

    cudaMemcpy(output.data, d_output, channel * sizeof(uchar)*input.rows*input.cols, cudaMemcpyDeviceToHost);

    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
}


__global__ void and_one_channel(uchar *d_input1, uchar *d_input2, int height, int width, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    auto add_pixels = [](uchar4 *a, uchar4 *b)
    {
        uchar m = a->x & b->x;
        uchar n = a->y & b->y;
        uchar p = a->z & b->z;
        uchar q = a->w & b->w;
        m = m > 255 ? 255 : m;
        n = m > 255 ? 255 : n;
        p = m > 255 ? 255 : p;
        q = m > 255 ? 255 : q;
        return make_uchar4(m, n, p, q);
    };

    // use blockDim.y*gridDim.y or blockDim.x*gridDim.x to retrieve elements, increasing ILP level, and memory accessing is coalesced.
    // visit this post, https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (uint i = row; i < height / 4; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (uint j = col; j < width; j += blockDim.x*gridDim.x)
        {
            // because use uchar, I just get 25% loading efficiency on global memory, a warp only request only 32 bytes, but ideally 128 bytes.
            // There's a post from CUDA blog, https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
            // as result, it benefit a lot performance improvement.
            uchar4 temp = reinterpret_cast<uchar4*>(d_input1)[i*width + j];
            uchar4 temp1 = reinterpret_cast<uchar4*>(d_input2)[i*width + j];
            reinterpret_cast<uchar4*>(d_output)[i*width + j] = add_pixels(&temp, &temp1);
        }
}


void cudaAND(const cv::Mat & input1, const cv::Mat & input2, cv::Mat & output)
{
    if (input1.channels() != input2.channels())return;
    if (input1.size() != input2.size())return;
    int channel = input1.channels();

    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 6);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input1.cols / (4 * block_size.x), input1.rows / (4 * block_size.y));

    uchar *d_input1, *d_input2, *d_output;
    cudaStream_t stream; cudaStreamCreate(&stream);

    cudaMalloc(&d_input1, channel * sizeof(uchar)*input1.cols*input1.rows);
    cudaMalloc(&d_input2, channel * sizeof(uchar)*input2.cols*input2.rows);
    cudaMalloc(&d_output, channel * sizeof(uchar)*input2.cols*input2.rows);
    cudaMemcpyAsync(d_input1, input1.data, channel * sizeof(uchar)*input1.rows*input1.cols, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_input2, input2.data, channel * sizeof(uchar)*input2.rows*input2.cols, cudaMemcpyHostToDevice, stream);

    switch (channel)
    {
    case 1:
        output = cv::Mat(input1.size(), input1.type(), cv::Scalar(0));
        and_one_channel <<<grid_size, block_size, 0, stream>>> (d_input1, d_input2, input1.rows, input1.cols, d_output);
        cudaDeviceSynchronize();
        break;
    case 3:
        output = cv::Mat(input1.size(), input1.type(), cv::Scalar(0, 0, 0));
        add_three_channel <<<grid_size, block_size, 0, stream>>> (d_input1, d_input2, input1.rows, input1.cols, d_output);
        cudaDeviceSynchronize();
        break;
    default:
        break;
    }

    cudaMemcpy(output.data, d_output, channel * sizeof(uchar)*input1.rows*input2.cols, cudaMemcpyDeviceToHost);

    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_input1)); CUDA_CALL(cudaFree(d_input2)); CUDA_CALL(cudaFree(d_output));
}