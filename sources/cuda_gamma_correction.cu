#include "../include/cuda_gamma_correction.h"


__device__ uchar4 gamma_handler(uchar4 pixels, float gamma)
{
    /*uchar p0 = __powf((float)pixels.x / 255.0f, gamma)*255.0f;
    uchar p1 = __powf((float)pixels.y / 255.0f, gamma)*255.0f;
    uchar p2 = __powf((float)pixels.z / 255.0f, gamma)*255.0f;
    uchar p3 = __powf((float)pixels.w / 255.0f, gamma)*255.0f;*/
    float t = fabsf(gamma - 1.0f);
    uchar p0 = __powf((float)pixels.x, gamma)*__powf(255.0f, t);
    uchar p1 = __powf((float)pixels.y, gamma)*__powf(255.0f, t);
    uchar p2 = __powf((float)pixels.z, gamma)*__powf(255.0f, t);
    uchar p3 = __powf((float)pixels.w, gamma)*__powf(255.0f, t);

    return make_uchar4(p0, p1, p2, p3);
}


template<int channels>
__global__ void gamma_correction(uchar *d_input, int height, int width, float gamma, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    for (int i = row; i < channels * height / 4; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            uchar4 p = reinterpret_cast<uchar4*>(d_input)[i*width + j];
            reinterpret_cast<uchar4*>(d_output)[i*width + j] = gamma_handler(p, gamma);
        }
}


void cudaGammaCorrection(const cv::Mat & input, cv::Mat & output, float gamma)
{
    int channels = input.channels();

    output = cv::Mat(input.size(), input.type(), cv::Scalar(0));
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
    case 1:  gamma_correction<1> <<<grid_size, block_size, 0, stream>>>(d_input, input.rows, input.cols, gamma, d_output); break;
    case 3:  gamma_correction<3> <<<grid_size, block_size, 0, stream>>>(d_input, input.rows, input.cols, gamma, d_output); break;
    default: break;
    }
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(output.data, d_output, sizeof(uchar)*output.cols*output.rows*channels, cudaMemcpyDeviceToHost));

    // resources releasing
    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
}