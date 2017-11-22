#include "..\include\cuda_merge.h"


__global__ void merge(uchar *d_ch0, uchar *d_ch1, uchar *d_ch2, int height, int width, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x;

    __shared__ uchar reshaped_smem[6][128 * 3];

    for (int i = row; i < height / 4; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (int j = col; j + threadIdx.x < width; j += blockDim.x*gridDim.x)
        {
            int in_index = i*width + j + threadIdx.x;
            int out_index = 3 * i*width + 3 * j + threadIdx.x;
            uchar4 ch0 = reinterpret_cast<uchar4*>(d_ch0)[in_index];
            uchar4 ch1 = reinterpret_cast<uchar4*>(d_ch1)[in_index];
            uchar4 ch2 = reinterpret_cast<uchar4*>(d_ch2)[in_index];

            reinterpret_cast<uchar3*>(reshaped_smem[threadIdx.y])[4 * threadIdx.x] = make_uchar3(ch0.x, ch1.x, ch2.x);
            reinterpret_cast<uchar3*>(reshaped_smem[threadIdx.y])[4 * threadIdx.x + 1] = make_uchar3(ch0.y, ch1.y, ch2.y);
            reinterpret_cast<uchar3*>(reshaped_smem[threadIdx.y])[4 * threadIdx.x + 2] = make_uchar3(ch0.z, ch1.z, ch2.z);
            reinterpret_cast<uchar3*>(reshaped_smem[threadIdx.y])[4 * threadIdx.x + 3] = make_uchar3(ch0.w, ch1.w, ch2.w);
            __syncthreads();

            reinterpret_cast<uchar4*>(d_output)[out_index] = reinterpret_cast<uchar4*>(reshaped_smem[threadIdx.y])[threadIdx.x];
            reinterpret_cast<uchar4*>(d_output)[out_index + 32] = reinterpret_cast<uchar4*>(reshaped_smem[threadIdx.y])[threadIdx.x + 32];
            reinterpret_cast<uchar4*>(d_output)[out_index + 64] = reinterpret_cast<uchar4*>(reshaped_smem[threadIdx.y])[threadIdx.x + 64];
        }
}


void cudaMerge(const std::vector<cv::Mat> & channels, cv::Mat & output)
{
    if (channels.size() == 1)
    {
        channels[0].copyTo(output);
        return;
    }
    output = cv::Mat(channels[0].size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cudaStream_t stream; CUDA_CALL(cudaStreamCreate(&stream));

    uchar*d_ch0, *d_ch1, *d_ch2, *d_output;
    CUDA_CALL(cudaMalloc(&d_output, output.channels() * sizeof(uchar)*output.rows*output.cols));

    CUDA_CALL(cudaMalloc(&d_ch0, sizeof(uchar)*output.rows*output.cols));
    CUDA_CALL(cudaMemcpyAsync(d_ch0, channels[0].data, sizeof(uchar)*output.rows*output.cols, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&d_ch1, sizeof(uchar)*output.rows*output.cols));
    CUDA_CALL(cudaMemcpyAsync(d_ch1, channels[1].data, sizeof(uchar)*output.rows*output.cols, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&d_ch2, sizeof(uchar)*output.rows*output.cols));
    CUDA_CALL(cudaMemcpyAsync(d_ch2, channels[2].data, sizeof(uchar)*output.rows*output.cols, cudaMemcpyHostToDevice, stream));

    dim3 block_size(THREAD_MULTIPLE, 6);
    // proper grid size will get the best performance, I tried on my laptop, cc 2.1, NVS4200m, devided into 4*4 grids will get the best performance
    // less threads do more things. But caring about register usage.
    dim3 grid_size(output.cols / (4 * block_size.x), output.rows / (4 * block_size.y));

    merge <<<grid_size, block_size, 0, stream>>> (d_ch0, d_ch1, d_ch2, output.rows, output.cols, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(output.data, d_output, output.channels() * sizeof(uchar)*output.rows*output.cols, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_ch0)); CUDA_CALL(cudaFree(d_ch1)); CUDA_CALL(cudaFree(d_ch2)); CUDA_CALL(cudaFree(d_output));
}