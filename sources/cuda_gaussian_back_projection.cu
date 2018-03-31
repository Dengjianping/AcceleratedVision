#include "../include/cuda_gaussian_back_projection.h"


__global__ void weight_of_channel(uchar *d_input, float *r_portion, float *g_portion, uint n)
{
    uint i = 3 * blockIdx.x*(blockDim.x * 2) + threadIdx.x; // input index
    uint j = blockIdx.x*(blockDim.x * 2) + threadIdx.x; // output index
    uint gridSize = 2 * blockDim.x * gridDim.x;

    __shared__ float smem[256 * 3 * 2];

    while (i < 3 * n)
    {
        uchar *pk = &d_input[i];
        smem[threadIdx.x] = *(pk + 0);
        smem[threadIdx.x + 256] = *(pk + 256);
        smem[threadIdx.x + 512] = *(pk + 512);
        smem[threadIdx.x + 768] = *(pk + 768);
        smem[threadIdx.x + 1024] = *(pk + 1024);
        smem[threadIdx.x + 1280] = *(pk + 1280);
        __syncthreads();

        float *p = &smem[3 * threadIdx.x];
        float rgb_sum0 = *(p + 0) + *(p + 1) + *(p + 2); // calculate the sum of channels to reduce registers usage
        float rgb_sum1 = *(p + 768) + *(p + 1 + 768) + *(p + 2 + 768);
        float r0 = __fdividef(*(p + 2), rgb_sum0); // use fast math, improve performance
        float g0 = __fdividef(*(p + 1), rgb_sum0);
        float r1 = __fdividef(*(p + 2 + 768), rgb_sum1);
        float g1 = __fdividef(*(p + 1 + 768), rgb_sum1);

        r_portion[j] = r0;
        g_portion[j] = g0;
        r_portion[j + 256] = r1;
        g_portion[j + 256] = g1;

        i += 3 * gridSize; // grid stride
        j += gridSize;
    }
}


template<uint blockSize>
__global__ void reduction_float(float *d_input, float *d_output, uint n)
{
    __shared__ float smem[256];

    uint i = blockIdx.x*(blockSize * 2) + threadIdx.x;
    uint gridSize = 2 * blockDim.x * gridDim.x;

    smem[threadIdx.x] = 0;
    while (i < n / 4)
    {
        // each block handle 256 * 4 *2 pxels, 256 threads in a block, which will read 256 * 4 pixels
        // and while reads another 1024 pixels, this will improve 3x performance than just handle 256 * 1 * 2
        float4 p0 = reinterpret_cast<float4*>(d_input)[i];
        float4 p1 = reinterpret_cast<float4*>(d_input)[i + blockSize];
        smem[threadIdx.x] += (p0.x + p1.x) + (p0.y + p1.y) + (p0.z + p1.z) + (p0.w + p1.w);
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 256)
    {
        if (threadIdx.x < 128)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 128];
        }
        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (threadIdx.x < 64) { smem[threadIdx.x] += smem[threadIdx.x + 64]; } __syncthreads();
    }

    if (threadIdx.x < 32)
    {
        if (blockSize >= 64)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 32];
        };
        __syncthreads();

        if (blockSize >= 32) smem[threadIdx.x] += smem[threadIdx.x + 16];
        __syncthreads();

        if (blockSize >= 16) smem[threadIdx.x] += smem[threadIdx.x + 8];
        __syncthreads();

        if (blockSize >= 8) smem[threadIdx.x] += smem[threadIdx.x + 4];
        __syncthreads();

        if (blockSize >= 4) smem[threadIdx.x] += smem[threadIdx.x + 2];
        __syncthreads();

        if (blockSize >= 2) smem[threadIdx.x] += smem[threadIdx.x + 1];
        __syncthreads();
    }

    if (threadIdx.x == 0) d_output[blockIdx.x] += smem[0]; // store result to global memory
}


template<uint blockSize>
__global__ void std_error(float *d_input, float mean, float *d_output, uint n)
{
    __shared__ float smem[256];

    uint i = blockIdx.x*(blockSize * 2) + threadIdx.x;
    uint gridSize = 2 * blockDim.x * gridDim.x;

    smem[threadIdx.x] = 0;
    while (i < n / 4)
    {
        // each block handle 256 * 4 *2 pxels, 256 threads in a block, which will read 256 * 4 pixels
        // and while reads another 1024 pixels, this will improve 3x performance than just handle 256 * 1 * 2
        float4 p0 = reinterpret_cast<float4*>(d_input)[i];
        float4 p1 = reinterpret_cast<float4*>(d_input)[i + blockSize];
        smem[threadIdx.x] += (powf(mean - p0.x, 2.0f) + powf(mean - p1.x, 2.0f)) + (powf(mean - p0.y, 2.0f) + powf(mean - p1.y, 2.0f)) + \
            (powf(mean - p0.z, 2.0f) + powf(mean - p1.z, 2.0f)) + (powf(mean - p0.w, 2.0f) + powf(mean - p1.w, 2.0f));
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 256)
    {
        if (threadIdx.x < 128)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 128];
        }
        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (threadIdx.x < 64) { smem[threadIdx.x] += smem[threadIdx.x + 64]; } __syncthreads();
    }

    if (threadIdx.x < 32)
    {
        if (blockSize >= 64)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 32];
        };
        __syncthreads();

        if (blockSize >= 32) smem[threadIdx.x] += smem[threadIdx.x + 16];
        __syncthreads();

        if (blockSize >= 16) smem[threadIdx.x] += smem[threadIdx.x + 8];
        __syncthreads();

        if (blockSize >= 8) smem[threadIdx.x] += smem[threadIdx.x + 4];
        __syncthreads();

        if (blockSize >= 4) smem[threadIdx.x] += smem[threadIdx.x + 2];
        __syncthreads();

        if (blockSize >= 2) smem[threadIdx.x] += smem[threadIdx.x + 1];
        __syncthreads();
    }

    if (threadIdx.x == 0) d_output[blockIdx.x] += smem[0]; // store result to global memory
}


__global__ void back_projection(uchar *d_input, float r_mean, float g_mean, float r_std, float g_std, float *d_output, uint n)
{
    uint i = 3 * blockIdx.x*(blockDim.x * 2) + threadIdx.x; // input index
    uint j = blockIdx.x*(blockDim.x * 2) + threadIdx.x; // output index
    uint gridSize = 2 * blockDim.x * gridDim.x;

    __shared__ float smem[256 * 3 * 2];

    float coeffiency = __fdividef(0.159154f, r_std*g_std);
    while (i < 3 * n)
    {
        uchar *pk = &d_input[i];
        smem[threadIdx.x] = *(pk + 0);
        smem[threadIdx.x + 256] = *(pk + 256);
        smem[threadIdx.x + 512] = *(pk + 512);
        smem[threadIdx.x + 768] = *(pk + 768);
        smem[threadIdx.x + 1024] = *(pk + 1024);
        smem[threadIdx.x + 1280] = *(pk + 1280);
        __syncthreads();

        float *p = &smem[3 * threadIdx.x];
        float rgb_sum0 = *(p + 0) + *(p + 1) + *(p + 2); // calculate the sum of channels to reduce registers usage
        float rgb_sum1 = *(p + 768) + *(p + 1 + 768) + *(p + 2 + 768);
        float r0 = __fdividef(*(p + 2), rgb_sum0); // use fast math, improve performance
        float g0 = __fdividef(*(p + 1), rgb_sum0);
        float r1 = __fdividef(*(p + 2 + 768), rgb_sum1);
        float g1 = __fdividef(*(p + 1 + 768), rgb_sum1);

        //float r_ref0 = rsqrtf(r_std*2.506628f) * expf(-powf((r0 - r_mean) / r_std, 2.0f) * 0.5f);
        //float r_ref1 = rsqrtf(r_std*2.506628f) * expf(-powf((r1 - r_mean) / r_std, 2.0f) * 0.5f);
        //float g_ref0 = rsqrtf(g_std*2.506628f) * expf(-powf((g0 - g_mean) / g_std, 2.0f) * 0.5f);
        //float g_ref1 = rsqrtf(g_std*2.506628f) * expf(-powf((g1 - g_mean) / g_std, 2.0f) * 0.5f);

        float r0_p = powf(__fdividef(r0 - r_mean, r_std), 2.0f);
        float g0_p = powf(__fdividef(g0 - g_mean, g_std), 2.0f);
        float r1_p = powf(__fdividef(r1 - r_mean, r_std), 2.0f);
        float g1_p = powf(__fdividef(g1 - g_mean, g_std), 2.0f);
        
        float r0_x_g0 = coeffiency * __expf(-(r0_p + g0_p)*0.5f); // merge r_ref0 * g_ref0
        float r1_x_g1 = coeffiency * __expf(-(r1_p + g1_p)*0.5f);

        d_output[j] = r0_x_g0;
        d_output[j + 256] = r1_x_g1;

        i += 3 * gridSize; // grid stride
        j += gridSize;
    }
}


void cudaGaussianBackProjection(const cv::Mat & input, const cv::Mat & model, cv::Mat & output)
{
    if (input.channels() != 3 && model.channels() != 3)return;
    int channels = input.channels();

    output = cv::Mat(input.size(), CV_32F);

    dim3 block_size(256, 1);
    // proper grid size will get the best performance, I tried on my laptop, cc 2.1, NVS4200m, devided into 4*4 grids will get the best performance
    // less threads do more things. But caring about register usage.
    dim3 grid_size(input.cols*input.rows / (128 * block_size.x), 1);
    dim3 model_grid_size(model.cols*model.rows / (4 * block_size.x), 1);

    cudaStream_t stream, stream1; cudaStreamCreate(&stream); cudaStreamCreate(&stream1);

    uchar *d_input; // allocate input data on device
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows*channels));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows*channels, cudaMemcpyHostToDevice, stream));

    float *d_output;
    CUDA_CALL(cudaMalloc(&d_output, sizeof(float)*input.cols*input.rows));

    uchar *model_input; // allocate model data on device
    CUDA_CALL(cudaMalloc(&model_input, sizeof(uchar)*model.cols*model.rows*channels));
    CUDA_CALL(cudaMemcpy(model_input, model.data, sizeof(uchar)*model.cols*model.rows*channels, cudaMemcpyHostToDevice));

    float *r_portion, *g_portion; // store each block of sum, r / (r + b + g), g / (r + b+g)
    CUDA_CALL(cudaMalloc(&r_portion, sizeof(float)*model.rows*model.cols));
    CUDA_CALL(cudaMalloc(&g_portion, sizeof(float)*model.rows*model.cols));

    float *r_d_sum, *g_d_sum; // store each block of sum
    CUDA_CALL(cudaMalloc(&r_d_sum, sizeof(float)*grid_size.x));
    CUDA_CALL(cudaMalloc(&g_d_sum, sizeof(float)*grid_size.x));
    CUDA_CALL(cudaMemset(r_d_sum, sizeof(float)*grid_size.x, 0));
    CUDA_CALL(cudaMemset(g_d_sum, sizeof(float)*grid_size.x, 0));

    float *r_h_sum = new float[grid_size.x];
    float *g_h_sum = new float[grid_size.x];

    float *r_d_std, *g_d_std; // store each block of sum
    CUDA_CALL(cudaMalloc(&r_d_std, sizeof(float)*grid_size.x));
    CUDA_CALL(cudaMalloc(&g_d_std, sizeof(float)*grid_size.x));
    CUDA_CALL(cudaMemset(r_d_std, sizeof(float)*grid_size.x, 0));
    CUDA_CALL(cudaMemset(g_d_std, sizeof(float)*grid_size.x, 0));

    float *r_h_std = new float[grid_size.x];
    float *g_h_std = new float[grid_size.x];

    weight_of_channel <<<model_grid_size, block_size, 0, stream>>> (model_input, r_portion, g_portion, model.rows * model.cols); // calculate the r, g channel weight

    reduction_float<256> <<<model_grid_size, block_size, 0, stream>>> (r_portion, r_d_sum, model.rows * model.cols);
    reduction_float<256> <<<model_grid_size, block_size, 0, stream1>>> (g_portion, g_d_sum, model.rows * model.cols);

    CUDA_CALL(cudaMemcpyAsync(r_h_sum, r_d_sum, sizeof(float)*grid_size.x, cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaMemcpyAsync(g_h_sum, g_d_sum, sizeof(float)*grid_size.x, cudaMemcpyDeviceToHost, stream1));

    for (int j = 1; j < grid_size.x; j++)
    {
        r_h_sum[0] += r_h_sum[j];
        g_h_sum[0] += g_h_sum[j];
    }
    float r_mean = r_h_sum[0] / ((float)model.rows * model.cols); // the mean value of each channel
    float g_mean = g_h_sum[0] / ((float)model.rows * model.cols);

    std_error<256> <<<model_grid_size, block_size, 0, stream>>> (r_portion, r_mean, r_d_std, model.rows * model.cols);
    std_error<256> <<<model_grid_size, block_size, 0, stream1>>> (g_portion, g_mean, g_d_std, model.rows * model.cols);

    CUDA_CALL(cudaMemcpyAsync(r_h_std, r_d_std, sizeof(float)*grid_size.x, cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaMemcpyAsync(g_h_std, g_d_std, sizeof(float)*grid_size.x, cudaMemcpyDeviceToHost, stream1));

    for (int j = 1; j < grid_size.x; j++)
    {
        r_h_std[0] += r_h_std[j];
        g_h_std[0] += g_h_std[j];
    }
    float r_std = sqrt(r_h_std[0] / ((float)model.rows * model.cols)); // the std value of each channel
    float g_std = sqrt(g_h_std[0] / ((float)model.rows * model.cols));

    back_projection <<<grid_size, block_size, 0, stream>>> (d_input, r_mean, g_mean, r_std, g_std, d_output, input.cols*input.rows);

    CUDA_CALL(cudaMemcpy(output.data, d_output, sizeof(float)*input.cols*input.rows, cudaMemcpyDeviceToHost));

    cv::normalize(output, output, 0, 255, cv::NORM_MINMAX); // normalize the result
    output.convertTo(output, CV_8U); // float to uchar

    input.copyTo(output, output); // use the result to mask the source

    CUDA_CALL(cudaStreamDestroy(stream)); CUDA_CALL(cudaStreamDestroy(stream1));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(r_portion)); CUDA_CALL(cudaFree(g_portion)); CUDA_CALL(cudaFree(g_d_sum));
    CUDA_CALL(cudaFree(r_d_std)); CUDA_CALL(cudaFree(g_d_std)); CUDA_CALL(cudaFree(d_output)); CUDA_CALL(cudaFree(r_d_sum)); CUDA_CALL(cudaFree(model_input));

    delete[] r_h_sum;
    delete[] g_h_sum;
    delete[] r_h_std;
    delete[] g_h_std;
}