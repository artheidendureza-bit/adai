#include "adaicpp/tensor.h"
#include "adaicpp/device.h"
#ifdef ADAICPP_ENABLE_CUDA
#include "adaicpp/cuda_kernels.h"
#endif
#include <stdexcept>
#include <cmath>
#include <random>
#include <algorithm>

namespace adaicpp {

Tensor::Tensor() 
    : size_(0), data_(nullptr), owns_data_(false), requires_grad_(false) {}

Tensor::Tensor(const std::vector<size_t>& shape, std::shared_ptr<Device> device)
    : shape_(shape), size_(0), data_(nullptr), owns_data_(true), requires_grad_(false) {
    
    if (!device) {
        device = DeviceManager::instance().get_cpu_device();
    }
    device_ = device;
    
    compute_strides();
    size_ = shape_.empty() ? 0 : strides_[0] * shape_[0];
    allocate_memory();
}

Tensor::Tensor(const std::vector<size_t>& shape, float value, std::shared_ptr<Device> device)
    : Tensor(shape, device) {
    fill(value);
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, std::shared_ptr<Device> device)
    : Tensor(shape, device) {
    if (data.size() != size_) {
        throw std::runtime_error("Data size does not match tensor shape");
    }
    
    if (device_->type() == DeviceType::CPU) {
        std::memcpy(data_, data.data(), size_ * sizeof(float));
    } else {
        device_->memcpy_host_to_device(data_, data.data(), size_ * sizeof(float));
    }
}

// Copy constructor
Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), strides_(other.strides_), size_(other.size_),
      device_(other.device_), data_(nullptr), owns_data_(true),
      requires_grad_(other.requires_grad_), backward_fn_(other.backward_fn_) {
    
    allocate_memory();
    
    if (size_ > 0) {
        if (device_->type() == DeviceType::CPU && other.device_->type() == DeviceType::CPU) {
            std::memcpy(data_, other.data_, size_ * sizeof(float));
        } else {
            // Handle device-to-device copy
            if (device_->type() == DeviceType::CPU) {
                device_->memcpy_device_to_host(data_, other.data_, size_ * sizeof(float));
            } else {
                device_->memcpy_host_to_device(data_, other.data_, size_ * sizeof(float));
            }
        }
    }
    
    if (other.grad_) {
        grad_ = std::make_shared<Tensor>(*other.grad_);
    }
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
      size_(other.size_), device_(std::move(other.device_)), data_(other.data_),
      owns_data_(other.owns_data_), requires_grad_(other.requires_grad_),
      grad_(std::move(other.grad_)), backward_fn_(std::move(other.backward_fn_)) {
    
    other.data_ = nullptr;
    other.size_ = 0;
    other.owns_data_ = false;
}

// Copy assignment
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        deallocate_memory();
        
        shape_ = other.shape_;
        strides_ = other.strides_;
        size_ = other.size_;
        device_ = other.device_;
        owns_data_ = true;
        requires_grad_ = other.requires_grad_;
        backward_fn_ = other.backward_fn_;
        
        allocate_memory();
        
        if (size_ > 0) {
            if (device_->type() == DeviceType::CPU && other.device_->type() == DeviceType::CPU) {
                std::memcpy(data_, other.data_, size_ * sizeof(float));
            } else {
                if (device_->type() == DeviceType::CPU) {
                    device_->memcpy_device_to_host(data_, other.data_, size_ * sizeof(float));
                } else {
                    device_->memcpy_host_to_device(data_, other.data_, size_ * sizeof(float));
                }
            }
        }
        
        if (other.grad_) {
            grad_ = std::make_shared<Tensor>(*other.grad_);
        } else {
            grad_.reset();
        }
    }
    return *this;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate_memory();
        
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        size_ = other.size_;
        device_ = std::move(other.device_);
        data_ = other.data_;
        owns_data_ = other.owns_data_;
        requires_grad_ = other.requires_grad_;
        grad_ = std::move(other.grad_);
        backward_fn_ = std::move(other.backward_fn_);
        
        other.data_ = nullptr;
        other.size_ = 0;
        other.owns_data_ = false;
    }
    return *this;
}

Tensor::~Tensor() {
    deallocate_memory();
}

// Data access
float* Tensor::data() {
    return data_;
}

const float* Tensor::data() const {
    return data_;
}

// Element access
float& Tensor::operator[](const std::vector<size_t>& indices) {
    if (indices.size() != shape_.size()) {
        throw std::runtime_error("Index dimensions do not match tensor dimensions");
    }
    
    size_t idx = compute_index(indices);
    
    if (device_->type() == DeviceType::CPU) {
        return data_[idx];
    } else {
        // For CUDA, we need to copy to host first
        static thread_local float host_value;
        device_->memcpy_device_to_host(&host_value, &data_[idx], sizeof(float));
        return host_value;
    }
}

const float& Tensor::operator[](const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::runtime_error("Index dimensions do not match tensor dimensions");
    }
    
    size_t idx = compute_index(indices);
    
    if (device_->type() == DeviceType::CPU) {
        return data_[idx];
    } else {
        // For CUDA, we need to copy to host first
        static thread_local float host_value;
        device_->memcpy_device_to_host(&host_value, &data_[idx], sizeof(float));
        return host_value;
    }
}

// Device operations
void Tensor::to_device(std::shared_ptr<Device> device) {
    if (device == device_) {
        return;
    }
    
    if (!owns_data_) {
        throw std::runtime_error("Cannot move non-owned tensor to different device");
    }
    
    void* new_data = device->allocate(size_ * sizeof(float));
    
    if (device->type() == DeviceType::CPU) {
        if (device_->type() == DeviceType::CPU) {
            std::memcpy(new_data, data_, size_ * sizeof(float));
        } else {
            device_->memcpy_device_to_host(new_data, data_, size_ * sizeof(float));
        }
    } else {
        if (device_->type() == DeviceType::CPU) {
            device->memcpy_host_to_device(new_data, data_, size_ * sizeof(float));
        } else {
            device->memcpy(new_data, data_, size_ * sizeof(float));
        }
    }
    
    device_->deallocate(data_);
    data_ = static_cast<float*>(new_data);
    device_ = device;
}

void Tensor::to_cpu() {
    to_device(DeviceManager::instance().get_cpu_device());
}

// Utility functions
void Tensor::fill(float value) {
    if (device_->type() == DeviceType::CPU) {
        std::fill(data_, data_ + size_, value);
    } else {
        // For CUDA, we'd need a kernel or copy to host, fill, copy back
        std::vector<float> host_data(size_, value);
        device_->memcpy_host_to_device(data_, host_data.data(), size_ * sizeof(float));
    }
}

void Tensor::zero() {
    fill(0.0f);
}

void Tensor::copy_from(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Cannot copy tensors with different shapes");
    }
    
    if (device_->type() == DeviceType::CPU && other.device_->type() == DeviceType::CPU) {
        std::memcpy(data_, other.data_, size_ * sizeof(float));
    } else {
        if (device_->type() == DeviceType::CPU) {
            device_->memcpy_device_to_host(data_, other.data_, size_ * sizeof(float));
        } else {
            device_->memcpy_host_to_device(data_, other.data_, size_ * sizeof(float));
        }
    }
}

bool Tensor::is_contiguous() const {
    if (shape_.empty()) {
        return true;
    }
    
    std::vector<size_t> expected_strides(shape_.size());
    expected_strides.back() = 1;
    for (int i = shape_.size() - 2; i >= 0; --i) {
        expected_strides[i] = expected_strides[i + 1] * shape_[i + 1];
    }
    
    return strides_ == expected_strides;
}

// Reshape and view operations
Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_size = 1;
    for (size_t dim : new_shape) {
        new_size *= dim;
    }
    
    if (new_size != size_) {
        throw std::runtime_error("Reshape size must match original size");
    }
    
    Tensor result(new_shape, device_);
    result.copy_from(*this);
    return result;
}

Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    return reshape(new_shape);
}

Tensor Tensor::transpose(const std::vector<size_t>& dims) const {
    if (dims.size() != shape_.size()) {
        throw std::runtime_error("Transpose dims must match tensor dimensions");
    }
    
    std::vector<size_t> new_shape(shape_.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        new_shape[i] = shape_[dims[i]];
    }
    
    Tensor result(new_shape, device_);
    
    // Simple transpose implementation for 2D
    if (shape_.size() == 2 && dims.size() == 2 && dims[0] == 1 && dims[1] == 0) {
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                size_t src_idx = i * shape_[1] + j;
                size_t dst_idx = j * shape_[0] + i;
                if (device_->type() == DeviceType::CPU) {
                    result.data_[dst_idx] = data_[src_idx];
                }
            }
        }
    } else {
        throw std::runtime_error("Only 2D transpose is currently implemented");
    }
    
    return result;
}

// Gradient tracking
void Tensor::zero_grad() {
    if (grad_) {
        grad_->zero();
    }
}

// Static methods
Tensor Tensor::zeros(const std::vector<size_t>& shape, std::shared_ptr<Device> device) {
    return Tensor(shape, 0.0f, device);
}

Tensor Tensor::ones(const std::vector<size_t>& shape, std::shared_ptr<Device> device) {
    return Tensor(shape, 1.0f, device);
}

Tensor Tensor::randn(const std::vector<size_t>& shape, std::shared_ptr<Device> device) {
    Tensor result(shape, device);
    
    if (device->type() == DeviceType::CPU) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < result.size_; ++i) {
            result.data_[i] = dist(gen);
        }
    } else {
        // For CUDA, generate on host then copy
        std::vector<float> host_data(result.size_);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < result.size_; ++i) {
            host_data[i] = dist(gen);
        }
        
        device->memcpy_host_to_device(result.data_, host_data.data(), result.size_ * sizeof(float));
    }
    
    return result;
}

Tensor Tensor::arange(float start, float stop, float step, std::shared_ptr<Device> device) {
    size_t size = static_cast<size_t>((stop - start) / step);
    std::vector<size_t> shape = {size};
    Tensor result(shape, device);
    
    if (device->type() == DeviceType::CPU) {
        for (size_t i = 0; i < size; ++i) {
            result.data_[i] = start + i * step;
        }
    } else {
        std::vector<float> host_data(size);
        for (size_t i = 0; i < size; ++i) {
            host_data[i] = start + i * step;
        }
        device->memcpy_host_to_device(result.data_, host_data.data(), size * sizeof(float));
    }
    
    return result;
}

// Private methods
void Tensor::compute_strides() {
    if (shape_.empty()) {
        strides_.clear();
        return;
    }
    
    strides_.resize(shape_.size());
    strides_.back() = 1;
    for (int i = shape_.size() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

size_t Tensor::compute_index(const std::vector<size_t>& indices) const {
    size_t idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        idx += indices[i] * strides_[i];
    }
    return idx;
}

void Tensor::allocate_memory() {
    if (size_ > 0) {
        data_ = static_cast<float*>(device_->allocate(size_ * sizeof(float)));
    }
}

void Tensor::deallocate_memory() {
    if (data_ && owns_data_) {
        device_->deallocate(data_);
    }
    data_ = nullptr;
}

// Tensor operations
Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes must match for addition");
    }
    
    Tensor result(a.shape(), a.device());
    
    if (a.device()->type() == DeviceType::CPU) {
        const float* a_data = a.data();
        const float* b_data = b.data();
        float* r_data = result.data();
        
        for (size_t i = 0; i < result.size(); ++i) {
            r_data[i] = a_data[i] + b_data[i];
        }
    } else {
#ifdef ADAICPP_ENABLE_CUDA
        cuda::launch_add(a.data(), b.data(), result.data(), result.size());
#else
        throw std::runtime_error("CUDA operations not enabled");
#endif
    }
    
    return result;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes must match for subtraction");
    }
    
    Tensor result(a.shape(), a.device());
    
    if (a.device()->type() == DeviceType::CPU) {
        const float* a_data = a.data();
        const float* b_data = b.data();
        float* r_data = result.data();
        
        for (size_t i = 0; i < result.size(); ++i) {
            r_data[i] = a_data[i] - b_data[i];
        }
    } else {
        throw std::runtime_error("CUDA operations not yet implemented");
    }
    
    return result;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes must match for multiplication");
    }
    
    Tensor result(a.shape(), a.device());
    
    if (a.device()->type() == DeviceType::CPU) {
        const float* a_data = a.data();
        const float* b_data = b.data();
        float* r_data = result.data();
        
        for (size_t i = 0; i < result.size(); ++i) {
            r_data[i] = a_data[i] * b_data[i];
        }
    } else {
        throw std::runtime_error("CUDA operations not yet implemented");
    }
    
    return result;
}

Tensor div(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes must match for division");
    }
    
    Tensor result(a.shape(), a.device());
    
    if (a.device()->type() == DeviceType::CPU) {
        const float* a_data = a.data();
        const float* b_data = b.data();
        float* r_data = result.data();
        
        for (size_t i = 0; i < result.size(); ++i) {
            r_data[i] = a_data[i] / b_data[i];
        }
    } else {
        throw std::runtime_error("CUDA operations not yet implemented");
    }
    
    return result;
}

Tensor relu(const Tensor& x) {
    Tensor result(x.shape(), x.device());
    
    if (x.device()->type() == DeviceType::CPU) {
        const float* x_data = x.data();
        float* r_data = result.data();
        
        for (size_t i = 0; i < result.size(); ++i) {
            r_data[i] = std::max(0.0f, x_data[i]);
        }
    } else {
#ifdef ADAICPP_ENABLE_CUDA
        cuda::launch_relu(x.data(), result.data(), result.size());
#else
        throw std::runtime_error("CUDA operations not enabled");
#endif
    }
    
    return result;
}

Tensor sigmoid(const Tensor& x) {
    Tensor result(x.shape(), x.device());
    
    if (x.device()->type() == DeviceType::CPU) {
        const float* x_data = x.data();
        float* r_data = result.data();
        
        for (size_t i = 0; i < result.size(); ++i) {
            r_data[i] = 1.0f / (1.0f + std::exp(-std::max(-500.0f, std::min(500.0f, x_data[i]))));
        }
    } else {
#ifdef ADAICPP_ENABLE_CUDA
        cuda::launch_sigmoid(x.data(), result.data(), result.size());
#else
        throw std::runtime_error("CUDA operations not enabled");
#endif
    }
    
    return result;
}

Tensor tanh(const Tensor& x) {
    Tensor result(x.shape(), x.device());
    
    if (x.device()->type() == DeviceType::CPU) {
        const float* x_data = x.data();
        float* r_data = result.data();
        
        for (size_t i = 0; i < result.size(); ++i) {
            r_data[i] = std::tanh(x_data[i]);
        }
    } else {
#ifdef ADAICPP_ENABLE_CUDA
        cuda::launch_tanh(x.data(), result.data(), result.size());
#else
        throw std::runtime_error("CUDA operations not enabled");
#endif
    }
    
    return result;
}

Tensor pow(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes must match for pow");
    }
    
    Tensor result(a.shape(), a.device());
    
    if (a.device()->type() == DeviceType::CPU) {
        const float* a_data = a.data();
        const float* b_data = b.data();
        float* r_data = result.data();
        
        for (size_t i = 0; i < result.size(); ++i) {
            r_data[i] = std::pow(a_data[i], b_data[i]);
        }
    } else {
        throw std::runtime_error("CUDA operations not yet implemented");
    }
    
    return result;
}

Tensor exp(const Tensor& x) {
    Tensor result(x.shape(), x.device());
    
    if (x.device()->type() == DeviceType::CPU) {
        const float* x_data = x.data();
        float* r_data = result.data();
        
        for (size_t i = 0; i < result.size(); ++i) {
            r_data[i] = std::exp(x_data[i]);
        }
    } else {
        throw std::runtime_error("CUDA operations not yet implemented");
    }
    
    return result;
}

Tensor log(const Tensor& x) {
    Tensor result(x.shape(), x.device());
    
    if (x.device()->type() == DeviceType::CPU) {
        const float* x_data = x.data();
        float* r_data = result.data();
        
        for (size_t i = 0; i < result.size(); ++i) {
            r_data[i] = std::log(x_data[i]);
        }
    } else {
        throw std::runtime_error("CUDA operations not yet implemented");
    }
    
    return result;
}

Tensor softmax(const Tensor& x) {
    if (x.ndim() == 0) {
        throw std::runtime_error("Cannot apply softmax to scalar");
    }
    
    Tensor result(x.shape(), x.device());
    
    if (x.device()->type() == DeviceType::CPU) {
        const float* x_data = x.data();
        float* r_data = result.data();
        
        size_t last_dim = x.shape().back();
        size_t outer_size = x.size() / last_dim;
        
        for (size_t i = 0; i < outer_size; ++i) {
            float max_val = x_data[i * last_dim];
            for (size_t j = 1; j < last_dim; ++j) {
                if (x_data[i * last_dim + j] > max_val) {
                    max_val = x_data[i * last_dim + j];
                }
            }
            
            float sum = 0.0f;
            for (size_t j = 0; j < last_dim; ++j) {
                r_data[i * last_dim + j] = std::exp(x_data[i * last_dim + j] - max_val);
                sum += r_data[i * last_dim + j];
            }
            
            for (size_t j = 0; j < last_dim; ++j) {
                r_data[i * last_dim + j] /= sum;
            }
        }
    } else {
        throw std::runtime_error("CUDA operations not yet implemented");
    }
    
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error("Matrix multiplication requires 2D tensors");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("Inner dimensions must match for matrix multiplication");
    }
    
    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t N = b.shape()[1];
    
    std::vector<size_t> result_shape = {M, N};
    Tensor result(result_shape, a.device());
    
    if (a.device()->type() == DeviceType::CPU) {
        const float* a_data = a.data();
        const float* b_data = b.data();
        float* r_data = result.data();
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += a_data[i * K + k] * b_data[k * N + j];
                }
                r_data[i * N + j] = sum;
            }
        }
    } else {
#ifdef ADAICPP_ENABLE_CUDA
        cuda::launch_matmul(a.data(), b.data(), result.data(), M, K, N);
#else
        throw std::runtime_error("CUDA operations not enabled");
#endif
    }
    
    return result;
}

Tensor sum(const Tensor& x, const std::vector<size_t>& axes, bool keepdims) {
    if (axes.empty()) {
        // Sum all elements
        Tensor result({1}, x.device());
        if (x.device()->type() == DeviceType::CPU) {
            const float* x_data = x.data();
            float sum_val = 0.0f;
            for (size_t i = 0; i < x.size(); ++i) {
                sum_val += x_data[i];
            }
            result.data()[0] = sum_val;
        }
        return result;
    }
    
    throw std::runtime_error("Sum with axes not yet implemented");
}

Tensor mean(const Tensor& x, const std::vector<size_t>& axes, bool keepdims) {
    Tensor s = sum(x, axes, keepdims);
    if (x.device()->type() == DeviceType::CPU) {
        s.data()[0] /= static_cast<float>(x.size());
    }
    return s;
}

Tensor max(const Tensor& x, const std::vector<size_t>& axes, bool keepdims) {
    if (axes.empty()) {
        Tensor result({1}, x.device());
        if (x.device()->type() == DeviceType::CPU) {
            const float* x_data = x.data();
            float max_val = x_data[0];
            for (size_t i = 1; i < x.size(); ++i) {
                if (x_data[i] > max_val) {
                    max_val = x_data[i];
                }
            }
            result.data()[0] = max_val;
        }
        return result;
    }
    
    throw std::runtime_error("Max with axes not yet implemented");
}

Tensor min(const Tensor& x, const std::vector<size_t>& axes, bool keepdims) {
    if (axes.empty()) {
        Tensor result({1}, x.device());
        if (x.device()->type() == DeviceType::CPU) {
            const float* x_data = x.data();
            float min_val = x_data[0];
            for (size_t i = 1; i < x.size(); ++i) {
                if (x_data[i] < min_val) {
                    min_val = x_data[i];
                }
            }
            result.data()[0] = min_val;
        }
        return result;
    }
    
    throw std::runtime_error("Min with axes not yet implemented");
}

Tensor transpose(const Tensor& x, const std::vector<size_t>& dims) {
    return x.transpose(dims);
}

Tensor pad(const Tensor& x, const std::vector<std::pair<size_t, size_t>>& padding, float value) {
    if (padding.size() != x.ndim()) {
        throw std::runtime_error("Padding must be specified for each dimension");
    }
    
    // Calculate new shape
    std::vector<size_t> new_shape = x.shape();
    for (size_t i = 0; i < padding.size(); ++i) {
        new_shape[i] += padding[i].first + padding[i].second;
    }
    
    Tensor result(new_shape, x.device());
    result.fill(value);
    
    if (x.device()->type() == DeviceType::CPU) {
        const float* x_data = x.data();
        float* r_data = result.data();
        
        // Copy data with padding
        std::vector<size_t> src_indices(x.ndim());
        std::vector<size_t> dst_indices(x.ndim());
        
        for (size_t i = 0; i < x.size(); ++i) {
            // Convert linear index to multi-dimensional indices
            size_t temp = i;
            for (int d = x.ndim() - 1; d >= 0; --d) {
                src_indices[d] = temp % x.shape()[d];
                temp /= x.shape()[d];
            }
            
            // Calculate destination indices with padding
            for (size_t d = 0; d < x.ndim(); ++d) {
                dst_indices[d] = src_indices[d] + padding[d].first;
            }
            
            // Calculate destination linear index
            size_t dst_idx = 0;
            for (size_t d = 0; d < x.ndim(); ++d) {
                dst_idx = dst_idx * new_shape[d] + dst_indices[d];
            }
            
            r_data[dst_idx] = x_data[i];
        }
    } else {
        throw std::runtime_error("CUDA operations not yet implemented");
    }
    
    return result;
}

Tensor conv2d(
    const Tensor& input, const Tensor& kernel,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    // Input shape: [batch, in_channels, in_height, in_width]
    // Kernel shape: [out_channels, in_channels, kernel_height, kernel_width]
    // Output shape: [batch, out_channels, out_height, out_width]
    
    if (input.ndim() != 4) {
        throw std::runtime_error("Input must be 4D tensor [batch, in_channels, height, width]");
    }
    if (kernel.ndim() != 4) {
        throw std::runtime_error("Kernel must be 4D tensor [out_channels, in_channels, height, width]");
    }
    
    int batch_size = input.shape()[0];
    int in_channels = input.shape()[1];
    int in_height = input.shape()[2];
    int in_width = input.shape()[3];
    int out_channels = kernel.shape()[0];
    int kernel_height = kernel.shape()[2];
    int kernel_width = kernel.shape()[3];
    
    int out_height = (in_height + 2 * padding_h - kernel_height) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - kernel_width) / stride_w + 1;
    
    std::vector<size_t> output_shape = {static_cast<size_t>(batch_size), static_cast<size_t>(out_channels), 
                                        static_cast<size_t>(out_height), static_cast<size_t>(out_width)};
    Tensor result(output_shape, input.device());
    
    if (input.device()->type() == DeviceType::CPU) {
        // CPU implementation
        const float* input_data = input.data();
        const float* kernel_data = kernel.data();
        float* output_data = result.data();
        
        for (int b = 0; b < batch_size; b++) {
            for (int oc = 0; oc < out_channels; oc++) {
                for (int oh = 0; oh < out_height; oh++) {
                    for (int ow = 0; ow < out_width; ow++) {
                        float sum = 0;
                        for (int ic = 0; ic < in_channels; ic++) {
                            for (int kh = 0; kh < kernel_height; kh++) {
                                for (int kw = 0; kw < kernel_width; kw++) {
                                    int ih = oh * stride_h - padding_h + kh;
                                    int iw = ow * stride_w - padding_w + kw;
                                    
                                    if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                        int input_idx = b * in_channels * in_height * in_width +
                                                       ic * in_height * in_width +
                                                       ih * in_width + iw;
                                        int kernel_idx = oc * in_channels * kernel_height * kernel_width +
                                                        ic * kernel_height * kernel_width +
                                                        kh * kernel_width + kw;
                                        sum += input_data[input_idx] * kernel_data[kernel_idx];
                                    }
                                }
                            }
                        }
                        int output_idx = b * out_channels * out_height * out_width +
                                       oc * out_height * out_width +
                                       oh * out_width + ow;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }
    } else {
#ifdef ADAICPP_ENABLE_CUDA
        cuda::launch_conv2d(
            input.data(), kernel.data(), result.data(),
            batch_size, in_channels, out_channels,
            in_height, in_width,
            kernel_height, kernel_width,
            stride_h, stride_w,
            padding_h, padding_w
        );
#else
        throw std::runtime_error("CUDA operations not enabled");
#endif
    }
    
    return result;
}

Tensor scalar_mul(const Tensor& a, float scalar) {
    Tensor result(a.shape(), a.device());
    
    if (a.device()->type() == DeviceType::CPU) {
        const float* a_data = a.data();
        float* r_data = result.data();
        
        for (size_t i = 0; i < a.size(); ++i) {
            r_data[i] = a_data[i] * scalar;
        }
    } else {
#ifdef ADAICPP_ENABLE_CUDA
        // For now, implement on CPU even if CUDA is enabled
        // TODO: Add CUDA kernel for scalar multiplication
        const float* a_data = a.data();
        float* r_data = result.data();
        
        for (size_t i = 0; i < a.size(); ++i) {
            r_data[i] = a_data[i] * scalar;
        }
#else
        throw std::runtime_error("CUDA operations not enabled");
#endif
    }
    
    return result;
}

} // namespace adaicpp