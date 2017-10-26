#include "..\include\cuda_split.h"


__global__ void split(uchar *d_input, int height, int width, uchar *r_ch, uchar *g_ch, uchar *b_ch)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x;
    __shared__ uchar smem[6][128 * 3];

    for (int i = row; i < height / 4; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            int out_index = i*width + j + threadIdx.x;
            int index = 3 * i*width + 3 * j + threadIdx.x;

            uchar4 p0 = reinterpret_cast<uchar4*>(d_input)[index + 0];
            uchar4 p1 = reinterpret_cast<uchar4*>(d_input)[index + 32];
            uchar4 p2 = reinterpret_cast<uchar4*>(d_input)[index + 64];

            reinterpret_cast<uchar4*>(smem[threadIdx.y])[threadIdx.x + 0] = p0;
            reinterpret_cast<uchar4*>(smem[threadIdx.y])[threadIdx.x + 32] = p1;
            reinterpret_cast<uchar4*>(smem[threadIdx.y])[threadIdx.x + 64] = p2;
            __syncthreads();

            int stride = 12 * threadIdx.x;
            p0 = make_uchar4(smem[threadIdx.y][stride + 0], smem[threadIdx.y][stride + 3], smem[threadIdx.y][stride + 6], \
                smem[threadIdx.y][stride + 9]);
            p1 = make_uchar4(smem[threadIdx.y][stride + 1], smem[threadIdx.y][stride + 4], smem[threadIdx.y][stride + 7], \
                smem[threadIdx.y][stride + 10]);
            p2 = make_uchar4(smem[threadIdx.y][stride + 2], smem[threadIdx.y][stride + 5], smem[threadIdx.y][stride + 8], \
                smem[threadIdx.y][stride + 11]);

            reinterpret_cast<uchar4*>(r_ch)[out_index] = p0;
            reinterpret_cast<uchar4*>(g_ch)[out_index] = p1;
            reinterpret_cast<uchar4*>(b_ch)[out_index] = p2;
        }
}


void cudaSplit(const cv::Mat & input, std::vector<cv::Mat> & channels)
{
    if (input.channels() == 1)
    {
        channels.push_back(input);
        return;
    }

    for (size_t i = 0; i < input.channels(); i++) {
        cv::Mat ch = cv::Mat(input.size(), CV_8U, cv::Scalar(0));
        channels.push_back(ch);
    }

    cudaStream_t stream; CUDA_CALL(cudaStreamCreate(&stream));

    uchar *d_input, *r_ch, *g_ch, *b_ch;
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar3)*input.rows*input.cols));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar3)*input.rows*input.cols, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&r_ch, sizeof(uchar)*input.rows*input.cols));
    CUDA_CALL(cudaMalloc(&g_ch, sizeof(uchar)*input.rows*input.cols));
    CUDA_CALL(cudaMalloc(&b_ch, sizeof(uchar)*input.rows*input.cols));

    dim3 block_size(THREAD_MULTIPLE, 6);
    // proper grid size will get the best performance, I tried on my laptop, cc 2.1, NVS4200m, devided into 4*4 grids will get the best performance
    // less threads do more things. But caring about register usage.
    dim3 grid_size(input.cols / (4 * block_size.x), input.rows / (4 * block_size.y));

    split <<<grid_size, block_size, 0, stream >>>(d_input, input.rows, input.cols, r_ch, g_ch, b_ch);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(channels[0].data, r_ch, sizeof(uchar)*input.rows*input.cols, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(channels[1].data, g_ch, sizeof(uchar)*input.rows*input.cols, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(channels[2].data, b_ch, sizeof(uchar)*input.rows*input.cols, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(r_ch)); CUDA_CALL(cudaFree(g_ch)); CUDA_CALL(cudaFree(b_ch)); CUDA_CALL(cudaFree(d_input));
}