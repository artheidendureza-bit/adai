#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace adaicpp {
namespace cuda {

template<typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

template<typename T, int BLOCK_SIZE>
__global__ void matmul_shared_kernel(const T* A, const T* B, T* C, int M, int K, int N) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    T sum = 0;
    
    for (int k = 0; k < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; k++) {
        if (row < M && k * BLOCK_SIZE + tx < K) {
            As[ty][tx] = A[row * K + k * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0;
        }
        
        if (k * BLOCK_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(k * BLOCK_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

template<typename T>
__global__ void conv2d_kernel(
    const T* input, const T* kernel, T* output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width,
    int kernel_height, int kernel_width,
    int out_height, int out_width,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    int batch_idx = blockIdx.z;
    int out_channel = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_channel >= out_channels || out_idx >= out_height * out_width) {
        return;
    }
    
    int out_h = out_idx / out_width;
    int out_w = out_idx % out_width;
    
    T sum = 0;
    
    for (int in_channel = 0; in_channel < in_channels; in_channel++) {
        for (int kh = 0; kh < kernel_height; kh++) {
            for (int kw = 0; kw < kernel_width; kw++) {
                int in_h = out_h * stride_h - padding_h + kh;
                int in_w = out_w * stride_w - padding_w + kw;
                
                if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                    int input_idx = batch_idx * in_channels * in_height * in_width +
                                   in_channel * in_height * in_width +
                                   in_h * in_width + in_w;
                    int kernel_idx = out_channel * in_channels * kernel_height * kernel_width +
                                    in_channel * kernel_height * kernel_width +
                                    kh * kernel_width + kw;
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
    }
    
    int output_idx = batch_idx * out_channels * out_height * out_width +
                     out_channel * out_height * out_width +
                     out_h * out_width + out_w;
    output[output_idx] = sum;
}

template<typename T>
__global__ void add_kernel(const T* A, const T* B, T* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

template<typename T>
__global__ void sub_kernel(const T* A, const T* B, T* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] - B[idx];
    }
}

template<typename T>
__global__ void mul_kernel(const T* A, const T* B, T* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

template<typename T>
__global__ void div_kernel(const T* A, const T* B, T* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] / B[idx];
    }
}

template<typename T>
__global__ void relu_kernel(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

template<typename T>
__global__ void sigmoid_kernel(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

template<typename T>
__global__ void tanh_kernel(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

// Backward operations
template<typename T>
__global__ void relu_backward_kernel(const T* input, const T* grad, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? grad[idx] : 0;
    }
}

template<typename T>
__global__ void sigmoid_backward_kernel(const T* output, const T* grad, T* input_grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T sig = output[idx];
        input_grad[idx] = grad[idx] * sig * (1.0f - sig);
    }
}

template<typename T>
__global__ void tanh_backward_kernel(const T* output, const T* grad, T* input_grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T t = output[idx];
        input_grad[idx] = grad[idx] * (1.0f - t * t);
    }
}

// Mixed precision conversion
__global__ void fp32_to_fp16_kernel(const float* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__ void fp16_to_fp32_kernel(const __half* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}

// Launch functions
void launch_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matmul_shared_kernel<float, BLOCK_SIZE><<<gridDim, blockDim>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}

void launch_conv2d(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width,
    int kernel_height, int kernel_width,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    int out_height = (in_height + 2 * padding_h - kernel_height) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - kernel_width) / stride_w + 1;
    
    int threads_per_block = 256;
    int total_output = out_height * out_width;
    int blocks_per_grid = (total_output + threads_per_block - 1) / threads_per_block;
    
    dim3 blockDim(threads_per_block);
    dim3 gridDim(blocks_per_grid, out_channels, batch_size);
    
    conv2d_kernel<<<gridDim, blockDim>>>(
        input, kernel, output,
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_height, kernel_width,
        out_height, out_width,
        stride_h, stride_w,
        padding_h, padding_w
    );
    cudaDeviceSynchronize();
}

void launch_add(const float* A, const float* B, float* C, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    
    add_kernel<<<blocks_per_grid, threads_per_block>>>(A, B, C, size);
    cudaDeviceSynchronize();
}

void launch_relu(const float* input, float* output, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    
    relu_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, size);
    cudaDeviceSynchronize();
}

void launch_sigmoid(const float* input, float* output, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    
    sigmoid_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, size);
    cudaDeviceSynchronize();
}

void launch_tanh(const float* input, float* output, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    
    tanh_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, size);
    cudaDeviceSynchronize();
}

void launch_relu_backward(const float* input, const float* grad, float* output, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    
    relu_backward_kernel<<<blocks_per_grid, threads_per_block>>>(input, grad, output, size);
    cudaDeviceSynchronize();
}

void launch_sigmoid_backward(const float* output, const float* grad, float* input_grad, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    
    sigmoid_backward_kernel<<<blocks_per_grid, threads_per_block>>>(output, grad, input_grad, size);
    cudaDeviceSynchronize();
}

void launch_tanh_backward(const float* output, const float* grad, float* input_grad, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    
    tanh_backward_kernel<<<blocks_per_grid, threads_per_block>>>(output, grad, input_grad, size);
    cudaDeviceSynchronize();
}

void launch_fp32_to_fp16(const float* input, __half* output, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    
    fp32_to_fp16_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, size);
    cudaDeviceSynchronize();
}

void launch_fp16_to_fp32(const __half* input, float* output, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    
    fp16_to_fp32_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, size);
    cudaDeviceSynchronize();
}

} // namespace cuda
} // namespace adaicpp