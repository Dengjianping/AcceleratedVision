#include "../include/cuda_adjust.h"


/*
this algorithm explanation
http://blog.sina.com.cn/s/blog_14d1511ee0102ww6s.html

threr's a equation here:

[low_in, high_in] -> (0.0, 1.0)
[low_out, high_out] -> (0.0, 1.0)

if input < 255 * low_in
    output = 255 * low_out
if input > 255 * high_in
    output = 255 * high_out
if 255 * low_in < input < 255 * high_in
    output = (high_out - low_out) / (high_in - low_in) * (input - 255 * low_in) + 255 * low_out
*/


__device__ uchar filter_pixel(uchar pixel, float low_in, float high_in, float low_out, float high_out)
{
    // if (pixel < low_in*255.0f)return 255.0f*low_out;
    // if (pixel > high_in*255.0f)return 255.0f*high_out;
    // else
    // {
    //     uchar out = (high_out - low_out) / (high_in - low_in)*(pixel - low_in*255.0f) + 255.0f*low_out;
    //     return out;
    // }

    // use this way, there's almost no divergence, kernel execution from 5.873ms to 4.876 while using a image size 2048 * 2048
    uchar low_cut = 255.0f*low_out*(pixel < low_in*255.0f);
    uchar high_cut = 255.0f*high_out*(pixel > high_in*255.0f);
    uchar between = ((high_out - low_out) / (high_in - low_in)*(pixel - low_in*255.0f) + 255.0f*low_out)*(pixel > low_in*255.0f&&pixel < high_in*255.0f);
    return low_cut + high_cut + between;
}


template<int channels>
__global__ void adjust(uchar *d_input, int height, int width, float low_in, float high_in, float low_out, float high_out, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    // use blockDim.y*gridDim.y or blockDim.x*gridDim.x to retrieve elements, increasing ILP level, and memory accessing is coalesced.
    // visit this post, https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int i = row; i < channels * height / 4; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            // because use uchar, I just get 25% loading efficiency on global memory, a warp only request only 32 bytes, but ideally 128 bytes.
            // There's a post from CUDA blog, https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
            // as result, it benefit a lot performance improvement.
            uchar4 p = reinterpret_cast<uchar4*>(d_input)[i*width + j];
            uchar f0 = filter_pixel(p.x, low_in, high_in, low_out, high_out);
            uchar f1 = filter_pixel(p.y, low_in, high_in, low_out, high_out);
            uchar f2 = filter_pixel(p.z, low_in, high_in, low_out, high_out);
            uchar f3 = filter_pixel(p.w, low_in, high_in, low_out, high_out);
            reinterpret_cast<uchar4*>(d_output)[i*width + j] = make_uchar4(f0, f1, f2, f3);
        }
}


void cudaAdjust(const cv::Mat & input, cv::Mat & output, float low_in, float high_in, float low_out, float high_out)
{
    int channels = input.channels();

    output = cv::Mat(input.size(), input.type(), cv::Scalar(255));
    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 6);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (4 * block_size.x), input.rows / (4 * block_size.y));

    uchar *d_input, *d_output;
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows*channels));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows*channels, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&d_output, sizeof(uchar)*input.cols*input.rows*channels));

    switch (channels)
    {
    case 1: adjust<1> <<<grid_size, block_size, 0, stream>>>(d_input, input.rows, input.cols, low_in, high_in, low_out, high_out, d_output); break;
    case 3: adjust<3> <<<grid_size, block_size, 0, stream>>>(d_input, input.rows, input.cols, low_in, high_in, low_out, high_out, d_output); break;
    default: break;
    }
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(output.data, d_output, sizeof(uchar)*output.cols*output.rows*channels, cudaMemcpyDeviceToHost));

    // resources releasing
    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
}