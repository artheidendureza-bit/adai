#pragma once

#include "adaicpp/device.h"
#include <vector>
#include <memory>
#include <functional>
#include <initializer_list>

namespace adaicpp {

class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<size_t>& shape, std::shared_ptr<Device> device = nullptr);
    Tensor(const std::vector<size_t>& shape, float value, std::shared_ptr<Device> device = nullptr);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, std::shared_ptr<Device> device = nullptr);
    
    // Copy and move
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    ~Tensor();
    
    // Accessors
    const std::vector<size_t>& shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return size_; }
    const std::vector<size_t>& strides() const { return strides_; }
    std::shared_ptr<Device> device() const { return device_; }
    
    // Data access
    float* data();
    const float* data() const;
    
    // Element access
    float& operator[](const std::vector<size_t>& indices);
    const float& operator[](const std::vector<size_t>& indices) const;
    
    // Device operations
    void to_device(std::shared_ptr<Device> device);
    void to_cpu();
    
    // Utility functions
    void fill(float value);
    void zero();
    void copy_from(const Tensor& other);
    bool is_contiguous() const;
    
    // Reshape and view operations
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor view(const std::vector<size_t>& new_shape) const;
    Tensor transpose(const std::vector<size_t>& dims) const;
    
    // Gradient tracking
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
    Tensor* grad() { return grad_.get(); }
    const Tensor* grad() const { return grad_.get(); }
    void set_grad(std::shared_ptr<Tensor> grad) { grad_ = grad; }
    void zero_grad();
    
    // Autograd context
    using BackwardFn = std::function<std::vector<std::shared_ptr<Tensor>>(const Tensor&)>;
    void set_backward_fn(BackwardFn fn) { backward_fn_ = fn; }
    const BackwardFn& backward_fn() const { return backward_fn_; }
    
    // Static methods
    static Tensor zeros(const std::vector<size_t>& shape, std::shared_ptr<Device> device = nullptr);
    static Tensor ones(const std::vector<size_t>& shape, std::shared_ptr<Device> device = nullptr);
    static Tensor randn(const std::vector<size_t>& shape, std::shared_ptr<Device> device = nullptr);
    static Tensor arange(float start, float stop, float step = 1.0f, std::shared_ptr<Device> device = nullptr);
    
private:
    void compute_strides();
    size_t compute_index(const std::vector<size_t>& indices) const;
    void allocate_memory();
    void deallocate_memory();
    
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t size_;
    std::shared_ptr<Device> device_;
    float* data_;
    bool owns_data_;
    
    // Autograd
    bool requires_grad_;
    std::shared_ptr<Tensor> grad_;
    BackwardFn backward_fn_;
};

// Tensor operations (element-wise)
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor pow(const Tensor& a, const Tensor& b);
Tensor exp(const Tensor& x);
Tensor log(const Tensor& x);

// Scalar operations
Tensor scalar_mul(const Tensor& a, float scalar);

Tensor relu(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor tanh(const Tensor& x);
Tensor softmax(const Tensor& x);

// Matrix operations
Tensor matmul(const Tensor& a, const Tensor& b);

// Convolution operations
Tensor conv2d(
    const Tensor& input, const Tensor& kernel,
    int stride_h = 1, int stride_w = 1,
    int padding_h = 0, int padding_w = 0
);

// Reduction operations
Tensor sum(const Tensor& x, const std::vector<size_t>& axes = {}, bool keepdims = false);
Tensor mean(const Tensor& x, const std::vector<size_t>& axes = {}, bool keepdims = false);
Tensor max(const Tensor& x, const std::vector<size_t>& axes = {}, bool keepdims = false);
Tensor min(const Tensor& x, const std::vector<size_t>& axes = {}, bool keepdims = false);

// Utility operations
Tensor transpose(const Tensor& x, const std::vector<size_t>& dims);
Tensor pad(const Tensor& x, const std::vector<std::pair<size_t, size_t>>& padding, float value = 0.0f);

} // namespace adaicpp
