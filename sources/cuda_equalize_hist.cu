#include "..\include\cuda_equalize_hist.h"

#define BINS 256

texture<uint, 1, cudaReadModeElementType> text1D;


__global__ void get_histogram(uchar *d_input, int height, int width, uint *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    for (int i = row; i < height; i += blockDim.y*gridDim.y)
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            atomicAdd(&d_output[d_input[i*width + j]], 1);
        }
}


__global__ void accumulate_histogram(uint *d_input, const int N, uint *d_output)
{
    __shared__ uint smem[BINS];
    __shared__ uint seme_accumulate[BINS];

    smem[threadIdx.x] = d_input[threadIdx.x];
    seme_accumulate[threadIdx.x] = 0;
    __syncthreads();

    if (threadIdx.x < N)
    {
        for (int i = 0; i <= threadIdx.x; i++)seme_accumulate[threadIdx.x] += smem[i];
        __syncthreads();

        uint sum = seme_accumulate[255];
        seme_accumulate[threadIdx.x] = 255 * seme_accumulate[threadIdx.x] / sum;
        __syncthreads();

        d_output[threadIdx.x] = seme_accumulate[threadIdx.x];
    }
}


__global__ void equalize_histogram(uchar *d_input, uint *accumulate, int height, int width, uchar *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    for (int i = row; i < height / 4; i += blockDim.y*gridDim.y)
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            // use texture memory is a better solution than share memory

            uchar4 p0 = reinterpret_cast<uchar4*>(d_input)[i*width + j];
            uchar x = tex1Dfetch(text1D, int(p0.x));
            uchar y = tex1Dfetch(text1D, int(p0.y));
            uchar z = tex1Dfetch(text1D, int(p0.z));
            uchar w = tex1Dfetch(text1D, int(p0.w));
            reinterpret_cast<uchar4*>(d_output)[i*width + j] = make_uchar4(x, y, z, w);
        }
}


void cudaEqualizeHist(const cv::Mat & input, cv::Mat & output)
{
    output = cv::Mat(input.size(), CV_8U, cv::Scalar(0));
    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 8);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (4 * block_size.x), input.rows / (4 * block_size.y));

    uchar *d_input, *d_output; 
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&d_output, sizeof(uchar)*input.cols*input.rows));

    uint *hist, *accumulate;
    CUDA_CALL(cudaMalloc(&hist, sizeof(uint) * BINS));
    CUDA_CALL(cudaMalloc(&accumulate, sizeof(uint) * BINS));
    cudaMemset(hist, 0, sizeof(uint) * BINS);
    cudaMemset(accumulate, 0, sizeof(uint) * BINS);

    text1D.filterMode = cudaFilterModePoint;
    text1D.addressMode[0] = cudaAddressModeWrap;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uint>();
    // bind texture

    // calling kernel
    get_histogram<<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, hist);

    accumulate_histogram<<<1, BINS, 0, stream>>> (hist, BINS, accumulate);
    CUDA_CALL(cudaBindTexture(NULL, &text1D, accumulate, &desc, BINS * sizeof(uint)));

    equalize_histogram<<<grid_size, block_size, 0, stream>>> (d_input, accumulate, input.rows, input.cols, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(output.data, d_output, sizeof(uchar)*output.cols*output.rows, cudaMemcpyDeviceToHost));

    // resources releasing
    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
    CUDA_CALL(cudaUnbindTexture(&text1D));
}