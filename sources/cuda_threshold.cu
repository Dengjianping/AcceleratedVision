#include "..\include\cuda_threshold.h"


__device__ void binarization(uchar4 *input, uchar thresh, uchar max_value)
{
    input->x = input->x < thresh ? 0 : max_value;
    input->y = input->y < thresh ? 0 : max_value;
    input->z = input->z < thresh ? 0 : max_value;
    input->w = input->w < thresh ? 0 : max_value;
}


__global__ void threshold(uchar *input, int height, int width, uchar thresh, uchar max_value, uchar *output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    // use blockDim.y*gridDim.y or blockDim.x*gridDim.x to retrieve elements, increasing ILP level, and memory accessing is coalesced.
    // visit this post, https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int i = row; i < height / 4; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            // because use uchar, I just get 25% loading efficiency on global memory, a warp only request only 32 bytes, but ideally 128 bytes.
            // There's a post from CUDA blog, https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
            // as result, it benefit a lot performance improvement.
            uchar4 p = reinterpret_cast<uchar4*>(input)[i*width + j];
            binarization(&p, thresh, max_value);
            reinterpret_cast<uchar4*>(output)[i*width + j] = p;
        }
}


//extern "C"
void cudaThreshold(const cv::Mat & input, uchar thresh, uchar max_value, cv::Mat & output)
{
    if (max_value > 255)max_value = 255;
    output = cv::Mat(input.size(), CV_8U, cv::Scalar(0));
    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 6);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (4 * block_size.x), input.rows / (4 * block_size.y));

    uchar *d_input, *d_output;
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&d_output, sizeof(uchar)*input.cols*input.rows));

    // calling kernel
    threshold <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, thresh, max_value, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(output.data, d_output, sizeof(uchar)*output.cols*output.rows, cudaMemcpyDeviceToHost));

    // resources releasing
    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
}