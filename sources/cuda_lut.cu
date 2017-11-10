#include "../include/cuda_lut.h"


texture<uchar, 1, cudaReadModeElementType> text1D;

__global__ void lut(uchar *d_input, int height, int width, uchar *d_table, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    for (int i = row; i < height / 4; i += blockDim.y*gridDim.y)
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            uchar4 p = reinterpret_cast<uchar4*>(d_input)[i*width + j];

            uchar x = tex1Dfetch(text1D, int(p.x));
            uchar y = tex1Dfetch(text1D, int(p.y));
            uchar z = tex1Dfetch(text1D, int(p.z));
            uchar w = tex1Dfetch(text1D, int(p.w));

            reinterpret_cast<uchar4*>(d_output)[i*width + j] = make_uchar4(x, y, z, w);
        }
}


void cudaLUT(const cv::Mat & input, const cv::Mat & table, cv::Mat & output)
{
    if (input.channels() != 1)return;
    output = cv::Mat(input.size(), input.type(), cv::Scalar(0));

    cudaStream_t stream; CUDA_CALL(cudaStreamCreate(&stream));

    uchar *d_input, *d_output;
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.rows*input.cols));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.rows*input.cols, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&d_output, sizeof(uchar)*output.rows*output.cols));

    uchar *d_table;
    CUDA_CALL(cudaMalloc(&d_table, sizeof(uchar)*table.rows*table.cols));
    CUDA_CALL(cudaMemcpyAsync(d_table, table.data, sizeof(uchar)*table.rows*table.cols, cudaMemcpyHostToDevice, stream));

    // I tried kinds of memory, like global memmory, cannot be accessed alignly, share memory, having bank conflict, constant memory, warp divergence
    // texture memory is the best solution I have found by now.
    // setup texture
    text1D.filterMode = cudaFilterModePoint;
    text1D.addressMode[0] = cudaAddressModeWrap;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar>();
    // bind texture
    CUDA_CALL(cudaBindTexture(NULL, &text1D, d_table, &desc, table.cols*table.rows * sizeof(uchar)));

    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 6);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (4 * block_size.x), input.rows / (4 * block_size.y));

    lut <<<grid_size, block_size, 0, stream>>>(d_input, input.rows, input.cols, d_table, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(output.data, d_output, sizeof(uchar)*input.rows*input.cols, cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output)); CUDA_CALL(cudaStreamDestroy(stream));

    CUDA_CALL(cudaUnbindTexture(&text1D));
}