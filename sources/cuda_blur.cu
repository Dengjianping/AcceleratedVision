#include "../include/cuda_blur.h"


static __constant__ float gaussian_mask[21*21];


__host__ __device__ float two_dim_gaussian(int x, int y, float theta) 
{
    float coeffient = 1.0f / (2.0f * PI*powf(theta, 2.0f));
    float powerIndex = -(powf(x, 2.0f) + powf(y, 2.0f)) / (2.0f * powf(theta, 2.0f));
    return coeffient*expf(powerIndex);
}


template<int RADIUS>
__global__ void gaussian_blur(uchar *d_input, int height, int width, uchar *d_output, float theta)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = 2 * blockDim.x*blockIdx.x + threadIdx.x; // each thread handle 2 pixels

    static __shared__ float smem[6 + 2 * RADIUS][64 + 2 * RADIUS];
    for (int k = row; k < height; k += blockDim.y*gridDim.y)
    {
        int index = k*width + col;
        smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS] = d_input[index];
        smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS + 32] = d_input[index + 32];
        //smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS + 64] = d_input[index + 64];
        //smem[threadIdx.y + RADIUS][threadIdx.x + RADIUS + 96] = d_input[index + 96];

        //if ((row >= RADIUS && row < height/4) && (col >= RADIUS && col < width))
        {
            // up and bottom row
            if (threadIdx.y < RADIUS)
            {
                int global_index = (k - RADIUS)*width + col;
                smem[threadIdx.y][threadIdx.x + RADIUS] = d_input[global_index];
                smem[threadIdx.y][threadIdx.x + RADIUS + 32] = d_input[global_index + 32];
                //smem[threadIdx.y][threadIdx.x + RADIUS + 64] = d_input[global_index + 64];
                //smem[threadIdx.y][threadIdx.x + RADIUS + 96] = d_input[global_index + 96];
            }
            if (threadIdx.y + RADIUS >= 6)
            {
                int global_index = (k + RADIUS)*width + col;
                smem[threadIdx.y + 2 * RADIUS][threadIdx.x + RADIUS] = d_input[global_index];
                smem[threadIdx.y + 2 * RADIUS][threadIdx.x + RADIUS + 32] = d_input[global_index + 32];
                //smem[threadIdx.y + 2 * RADIUS][threadIdx.x + RADIUS + 64] = d_input[global_index + 64];
                //smem[threadIdx.y + 2 * RADIUS][threadIdx.x + RADIUS + 96] = d_input[global_index + 96];
            }
            // left and right column
            if (threadIdx.x < RADIUS)
                smem[threadIdx.y + RADIUS][threadIdx.x] = d_input[k*width + (col - RADIUS)];
            if (threadIdx.x + RADIUS >= 32)
                smem[threadIdx.y + RADIUS][threadIdx.x + 2 * RADIUS + 32] = d_input[k*width + col + RADIUS + 32];

            // load upper-left corner
            if (threadIdx.x < RADIUS && threadIdx.y < RADIUS)
                smem[threadIdx.y][threadIdx.x] = d_input[(k - RADIUS)*width + (col - RADIUS)];

            // load upper-right
            if (threadIdx.x + RADIUS >= 32 && threadIdx.y < RADIUS)
                smem[threadIdx.y][threadIdx.x + 2 * RADIUS + 32] = d_input[(k - RADIUS)*width + (col + RADIUS + 32)];

            // bottom-left
            if (threadIdx.y + RADIUS >= 6 && threadIdx.x < RADIUS)
                smem[threadIdx.y + 2 * RADIUS][threadIdx.x] = d_input[(k + RADIUS)*width + (col - RADIUS)];

            // bottom-right
            if (threadIdx.y + RADIUS >= 6 && threadIdx.x + RADIUS >= 32)
                smem[threadIdx.y + 2 * RADIUS][threadIdx.x + 2 * RADIUS + 32] = d_input[(k + RADIUS)*width + (col + RADIUS + 32)];
        }
        __syncthreads();

        float sum = 0.0f, sum_32 = 0.0f; //sum_64 = 0.0f, sum_96 = 0.0f;
        for (int i = -RADIUS; i <= RADIUS; i++)
            for (int j = -RADIUS; j <= RADIUS; j++)
            {
                sum += gaussian_mask[(RADIUS + i)*(2 * RADIUS + 1) + (RADIUS + j)] * smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j];
                sum_32 += gaussian_mask[(RADIUS + i)*(2 * RADIUS + 1) + (RADIUS + j)] * smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 32];
                //sum_64 += gaussian_mask4[RADIUS + i][RADIUS + j] * smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 64];
                //sum_96 += gaussian_mask4[RADIUS + i][RADIUS + j] * smem[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j + 96];
            }

        d_output[index] = (uchar)sum > 255?255 : (uchar)sum;
        d_output[index + 32] = (uchar)sum_32 > 255?255 : (uchar)sum_32;
        //d_output[index + 32] = ((int)sum_32 >> 8);
        //d_output[index + 64] = ((int)sum_64>>8);
        //d_output[index + 96] = ((int)sum_96>>8);
    }
}


template<int RADIUS>
__global__ void gaussian_blur_rgb(uchar *d_input, int height, int width, uchar *d_output, float theta)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x;

    static __shared__ float smem[4 + 2 * RADIUS][32 * 3 + 3 * 2 * RADIUS];
    static __shared__ float blurred[4][32 * 3];
    for (int k = row; k < height; k += blockDim.y*gridDim.y)
        for (int p = col; p < width; p += blockDim.x*gridDim.x)
        {
            int index = 3 * k*width + 3 * p + threadIdx.x;
            smem[threadIdx.y + RADIUS][threadIdx.x + 3 * RADIUS] = d_input[index];
            smem[threadIdx.y + RADIUS][threadIdx.x + 3 * RADIUS + 32] = d_input[index + 32];
            smem[threadIdx.y + RADIUS][threadIdx.x + 3 * RADIUS + 64] = d_input[index + 64];

            //if ((row >= RADIUS && row < height/4) && (col >= RADIUS && col < width))
            {
                // up and bottom row
                if (threadIdx.y < RADIUS)
                {
                    int global_index = 3 * (k - RADIUS)*width + 3 * p + threadIdx.x;
                    smem[threadIdx.y][threadIdx.x + 3 * RADIUS] = d_input[global_index];
                    smem[threadIdx.y][threadIdx.x + 3 * RADIUS + 32] = d_input[global_index + 32];
                    smem[threadIdx.y][threadIdx.x + 3 * RADIUS + 64] = d_input[global_index + 64];
                    //smem[threadIdx.y][threadIdx.x + RADIUS + 64] = d_input[global_index + 64];
                    //smem[threadIdx.y][threadIdx.x + RADIUS + 96] = d_input[global_index + 96];
                }
                if (threadIdx.y + RADIUS >= 6)
                {
                    int global_index = 3 * (k + RADIUS)*width + 3 * p + threadIdx.x;
                    smem[threadIdx.y + RADIUS][threadIdx.x + 3 * RADIUS] = d_input[global_index];
                    smem[threadIdx.y + RADIUS][threadIdx.x + 3 * RADIUS + 32] = d_input[global_index + 32];
                    smem[threadIdx.y + RADIUS][threadIdx.x + 3 * RADIUS + 64] = d_input[global_index + 64];
                    //smem[threadIdx.y + 2 * RADIUS][threadIdx.x + RADIUS + 96] = d_input[global_index + 96];
                }
                // left and right column
                if (threadIdx.x < RADIUS)
                {
                    smem[threadIdx.y + RADIUS][threadIdx.x] = d_input[k*width + (col - RADIUS)];
                    smem[threadIdx.y + RADIUS][threadIdx.x + 1] = d_input[k*width + (col - RADIUS) + 1];
                    smem[threadIdx.y + RADIUS][threadIdx.x + 2] = d_input[k*width + (col - RADIUS) + 2];
                }
                if (threadIdx.x + RADIUS >= 32)
                {
                    int global_index = 3 * k*width + 3 * p + +threadIdx.x + RADIUS + 96;
                    smem[threadIdx.y + RADIUS][threadIdx.x + 3 * RADIUS + 96] = d_input[global_index];
                    smem[threadIdx.y + RADIUS][threadIdx.x + 3 * RADIUS + 96 + 1] = d_input[global_index + 1];
                    smem[threadIdx.y + RADIUS][threadIdx.x + 3 * RADIUS + 96 + 2] = d_input[global_index + 2];
                }

                // load upper-left corner
                if (threadIdx.x < RADIUS && threadIdx.y < RADIUS)
                    smem[threadIdx.y][threadIdx.x] = d_input[(k - RADIUS)*width + (col - RADIUS)];

                // load upper-right
                if (threadIdx.x + RADIUS >= 32 && threadIdx.y < RADIUS)
                    smem[threadIdx.y][threadIdx.x + 2 * RADIUS + 32] = d_input[(k - RADIUS)*width + (col + RADIUS + 32)];

                // bottom-left
                if (threadIdx.y + RADIUS >= 6 && threadIdx.x < RADIUS)
                    smem[threadIdx.y + 2 * RADIUS][threadIdx.x] = d_input[(k + RADIUS)*width + (col - RADIUS)];

                // bottom-right
                if (threadIdx.y + RADIUS >= 6 && threadIdx.x + RADIUS >= 32)
                    smem[threadIdx.y + 2 * RADIUS][threadIdx.x + 2 * RADIUS + 32] = d_input[(k + RADIUS)*width + (col + RADIUS + 32)];
            }
            __syncthreads();

            float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
            for (int i = -RADIUS; i <= RADIUS; i++)
                for (int j = -RADIUS; j <= RADIUS; j++)
                {
                    //r_sum += gaussian_mask3[RADIUS + i][RADIUS + j] * smem[threadIdx.y + RADIUS - i][3 * (threadIdx.x + RADIUS - j)];
                    //g_sum += gaussian_mask3[RADIUS + i][RADIUS + j] * smem[threadIdx.y + RADIUS - i][3 * (threadIdx.x + RADIUS - j) + 1];
                    //b_sum += gaussian_mask3[RADIUS + i][RADIUS + j] * smem[threadIdx.y + RADIUS - i][3 * (threadIdx.x + RADIUS - j) + 2];

                    r_sum = fmaf(gaussian_mask[(RADIUS + i)*(2*RADIUS+1) + (RADIUS + j)], smem[threadIdx.y + RADIUS - i][3 * (threadIdx.x + RADIUS - j)], r_sum);
                    g_sum = fmaf(gaussian_mask[(RADIUS + i)*(2 * RADIUS + 1) + (RADIUS + j)], smem[threadIdx.y + RADIUS - i][3 * (threadIdx.x + RADIUS - j) + 1], g_sum);
                    b_sum = fmaf(gaussian_mask[(RADIUS + i)*(2 * RADIUS + 1) + (RADIUS + j)], smem[threadIdx.y + RADIUS - i][3 * (threadIdx.x + RADIUS - j) + 2], b_sum);
                }

            blurred[threadIdx.y][3 * threadIdx.x] = __fdividef(r_sum, 256.0f);
            blurred[threadIdx.y][3 * threadIdx.x + 1] = __fdividef(g_sum, 256.0f);
            blurred[threadIdx.y][3 * threadIdx.x + 2] = __fdividef(b_sum, 256.0f);
            __syncthreads();

            d_output[index] = blurred[threadIdx.y][threadIdx.x];
            d_output[index + 32] = blurred[threadIdx.y][threadIdx.x + 32];
            d_output[index + 64] = blurred[threadIdx.y][threadIdx.x + 64];
    }
}


void cudaGaussianBlur(const cv::Mat & input, cv::Mat & output, int radius, float theta)
{
    int diameter = 2 * radius + 1;
    float *host_gaussian_mask = new float[diameter*diameter]; float gaussian_sum = 0.0f;
    for (int i = 0; i < diameter; i++)
        for (int j = 0; j < diameter; j++)
        {
            host_gaussian_mask[i*diameter + j] = two_dim_gaussian(i - 3, 3 - j, 1.0f);
            gaussian_sum += host_gaussian_mask[i*diameter + j];
        }

    //normalization
    for (int i = 0; i < diameter; i++)
        for (int j = 0; j < diameter; j++)
        {
            host_gaussian_mask[i*diameter + j] /= gaussian_sum;
        }
    cudaMemcpyToSymbol(gaussian_mask, host_gaussian_mask, sizeof(float)*diameter*diameter, 0, cudaMemcpyHostToDevice);
    
    int channel = input.channels();
    output = cv::Mat(input.size(), input.type(), cv::Scalar(0));

    cudaStream_t stream; CUDA_CALL(cudaStreamCreate(&stream));

    uchar *d_input, *d_output;
    cudaMalloc(&d_input, channel * sizeof(uchar)*input.rows*input.cols);
    cudaMemcpyAsync(d_input, input.data, channel * sizeof(uchar)*input.rows*input.cols, cudaMemcpyHostToDevice, stream);
    cudaMalloc(&d_output, channel * sizeof(uchar)*input.rows*input.cols);

    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 6);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (2 * block_size.x), input.rows / (4 * block_size.y));

    switch (radius)
    {
    case 1:
        gaussian_blur<1> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output, theta);
        cudaDeviceSynchronize();
        break;
    case 2:
        gaussian_blur<2> <<<grid_size, block_size, 0, stream>> > (d_input, input.rows, input.cols, d_output, theta);
        cudaDeviceSynchronize();
        break;
    case 3:
        gaussian_blur<3> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output, theta);
        cudaDeviceSynchronize();
        break;
    case 4:
        gaussian_blur<4> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output, theta);
        cudaDeviceSynchronize();
        break;
    case 5:
        std::cout << radius << std::endl;
        gaussian_blur<5> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output, theta);
        cudaDeviceSynchronize();
        break;
    default:
        input.copyTo(output); 
        break;
    }

    cudaMemcpyAsync(output.data, d_output, channel * sizeof(uchar)*input.rows*input.cols, cudaMemcpyDeviceToHost, stream);
    cudaFree(d_input); cudaFree(d_output); cudaStreamDestroy(stream);
    delete[] host_gaussian_mask;
}


void cudaGaussianBlurRGB(const cv::Mat & input, cv::Mat & output, int radius, float theta)
{
    //if (input.channels() != 1)return;
    output = cv::Mat(input.size(), input.type(), cv::Scalar(0, 0, 0));

    cudaStream_t stream; CUDA_CALL(cudaStreamCreate(&stream));

    uchar *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(uchar3)*input.rows*input.cols);
    cudaMemcpyAsync(d_input, input.data, sizeof(uchar3)*input.rows*input.cols, cudaMemcpyHostToDevice, stream);
    cudaMalloc(&d_output, sizeof(uchar3)*input.rows*input.cols);

    // define block size and
    dim3 block_size(THREAD_MULTIPLE, 4);
    // divide the image into 16 grids, smaller grid do more things, improve performance a lot.
    dim3 grid_size(input.cols / (2 * block_size.x), input.rows / (4 * block_size.y));

    switch (radius)
    {
    case 1:
        gaussian_blur<1> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output, theta);
        cudaDeviceSynchronize();
        break;
    case 2:
        gaussian_blur<2> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output, theta);
        cudaDeviceSynchronize();
        break;
    case 3:
        gaussian_blur_rgb<3> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output, theta);
        cudaDeviceSynchronize();
        break;
    case 4:
        gaussian_blur<4> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output, theta);
        cudaDeviceSynchronize();
        break;
    case 5:
        gaussian_blur<5> <<<grid_size, block_size, 0, stream>>> (d_input, input.rows, input.cols, d_output, theta);
        cudaDeviceSynchronize();
        break;
    default:
        input.copyTo(output);
        break;
    }

    cudaMemcpyAsync(output.data, d_output, sizeof(uchar3)*input.rows*input.cols, cudaMemcpyDeviceToHost, stream);
    cudaFree(d_input); cudaFree(d_output); cudaStreamDestroy(stream);
}