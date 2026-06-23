#pragma once

#ifdef ADAICPP_ENABLE_CUDA
#include <cuda_fp16.h>

namespace adaicpp {
namespace cuda {

// Matrix multiplication
void launch_matmul(const float* A, const float* B, float* C, int M, int K, int N);

// 2D Convolution
void launch_conv2d(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width,
    int kernel_height, int kernel_width,
    int stride_h, int stride_w,
    int padding_h, int padding_w
);

// Element-wise operations
void launch_add(const float* A, const float* B, float* C, int size);
void launch_relu(const float* input, float* output, int size);
void launch_sigmoid(const float* input, float* output, int size);
void launch_tanh(const float* input, float* output, int size);

// Backward operations
void launch_relu_backward(const float* input, const float* grad, float* output, int size);
void launch_sigmoid_backward(const float* output, const float* grad, float* input_grad, int size);
void launch_tanh_backward(const float* output, const float* grad, float* input_grad, int size);

// Mixed precision conversion
void launch_fp32_to_fp16(const float* input, __half* output, int size);
void launch_fp16_to_fp32(const __half* input, float* output, int size);

} // namespace cuda
} // namespace adaicpp

#endif // ADAICPP_ENABLE_CUDA
