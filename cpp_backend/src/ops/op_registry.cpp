#include "adaicpp/op_registry.h"
#include "adaicpp/tensor.h"
#include "adaicpp/device.h"
#ifdef ADAICPP_ENABLE_CUDA
#include "adaicpp/cuda_kernels.h"
#endif
#include <stdexcept>
#include <string>

namespace adaicpp {

// OpRegistry Implementation
OpRegistry& OpRegistry::instance() {
    static OpRegistry instance;
    return instance;
}

void OpRegistry::register_op(const std::string& name, std::shared_ptr<GraphOp> op) {
    ops_[name] = op;
}

std::shared_ptr<GraphOp> OpRegistry::get_op(const std::string& name) {
    auto it = ops_.find(name);
    if (it == ops_.end()) {
        throw std::runtime_error("Operation '" + name + "' not found in registry");
    }
    return it->second;
}

bool OpRegistry::has_op(const std::string& name) const {
    return ops_.find(name) != ops_.end();
}

OpForwardFn OpRegistry::get_forward_fn(const std::string& name) {
    auto op = get_op(name);
    return op->forward_fn();
}

OpBackwardFn OpRegistry::get_backward_fn(const std::string& name) {
    auto op = get_op(name);
    return op->backward_fn();
}

// Register standard operations
void register_standard_operations() {
    auto& registry = OpRegistry::instance();
    
    // Register add
    registry.register_op("add", std::make_shared<GraphOp>(
        "add",
        add_forward,
        add_backward
    ));
    
    // Register mul
    registry.register_op("mul", std::make_shared<GraphOp>(
        "mul",
        mul_forward,
        mul_backward
    ));
    
    // Register matmul
    registry.register_op("matmul", std::make_shared<GraphOp>(
        "matmul",
        matmul_forward,
        matmul_backward
    ));
    
    // Register relu
    registry.register_op("relu", std::make_shared<GraphOp>(
        "relu",
        relu_forward,
        relu_backward
    ));
    
    // Register sigmoid
    registry.register_op("sigmoid", std::make_shared<GraphOp>(
        "sigmoid",
        sigmoid_forward,
        sigmoid_backward
    ));
    
    // Register tanh
    registry.register_op("tanh", std::make_shared<GraphOp>(
        "tanh",
        tanh_forward,
        tanh_backward
    ));
    
    // Register pow
    registry.register_op("pow", std::make_shared<GraphOp>(
        "pow",
        pow_forward,
        pow_backward
    ));
    
    // Register exp
    registry.register_op("exp", std::make_shared<GraphOp>(
        "exp",
        exp_forward,
        exp_backward
    ));
    
    // Register log
    registry.register_op("log", std::make_shared<GraphOp>(
        "log",
        log_forward,
        log_backward
    ));
    
    // Register softmax
    registry.register_op("softmax", std::make_shared<GraphOp>(
        "softmax",
        softmax_forward,
        softmax_backward
    ));
}

// Standard operation implementations
std::shared_ptr<Tensor> add_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 2) {
        throw std::runtime_error("Add operation requires exactly 2 inputs");
    }
    return std::make_shared<Tensor>(add(*inputs[0], *inputs[1]));
}

std::vector<std::shared_ptr<Tensor>> add_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs) {
    (void)inputs;  // Unused in simple implementation
    // Gradient of add is identity for both inputs (with broadcasting support)
    // For simplicity, we assume no broadcasting for now
    return {out_grad, out_grad};
}

std::shared_ptr<Tensor> mul_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 2) {
        throw std::runtime_error("Mul operation requires exactly 2 inputs");
    }
    return std::make_shared<Tensor>(mul(*inputs[0], *inputs[1]));
}

std::vector<std::shared_ptr<Tensor>> mul_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs) {
    // Gradient of mul: d/dx(x*y) = y, d/dy(x*y) = x
    auto dx = std::make_shared<Tensor>(mul(*out_grad, *inputs[1]));
    auto dy = std::make_shared<Tensor>(mul(*out_grad, *inputs[0]));
    return {dx, dy};
}

std::shared_ptr<Tensor> matmul_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 2) {
        throw std::runtime_error("Matmul operation requires exactly 2 inputs");
    }
    return std::make_shared<Tensor>(matmul(*inputs[0], *inputs[1]));
}

std::vector<std::shared_ptr<Tensor>> matmul_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs) {
    // Gradient of matmul: d/dx(A*B) = d_out * B^T, d/dB(A*B) = A^T * d_out
    auto& A = inputs[0];
    auto& B = inputs[1];
    
    // Compute dA = out_grad * B^T
    auto B_T = std::make_shared<Tensor>(transpose(*B, {1, 0}));
    auto dA = std::make_shared<Tensor>(matmul(*out_grad, *B_T));
    
    // Compute dB = A^T * out_grad
    auto A_T = std::make_shared<Tensor>(transpose(*A, {1, 0}));
    auto dB = std::make_shared<Tensor>(matmul(*A_T, *out_grad));
    
    return {dA, dB};
}

std::shared_ptr<Tensor> relu_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("ReLU operation requires exactly 1 input");
    }
    return std::make_shared<Tensor>(relu(*inputs[0]));
}

std::vector<std::shared_ptr<Tensor>> relu_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs) {
    // Gradient of ReLU: 1 if x > 0, 0 otherwise
    auto& x = inputs[0];
    
    // Create mask where x > 0
    auto mask = std::make_shared<Tensor>(Tensor::zeros(x->shape(), x->device()));
    if (x->device()->type() == DeviceType::CPU) {
        const float* x_data = x->data();
        float* mask_data = mask->data();
        for (size_t i = 0; i < x->size(); ++i) {
            mask_data[i] = x_data[i] > 0.0f ? 1.0f : 0.0f;
        }
    } else {
#ifdef ADAICPP_ENABLE_CUDA
        cuda::launch_relu_backward(x->data(), out_grad->data(), mask->data(), x->size());
#else
        throw std::runtime_error("CUDA operations not enabled");
#endif
    }
    
    auto dx = std::make_shared<Tensor>(mul(*out_grad, *mask));
    return {dx};
}

std::shared_ptr<Tensor> sigmoid_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("Sigmoid operation requires exactly 1 input");
    }
    return std::make_shared<Tensor>(sigmoid(*inputs[0]));
}

std::vector<std::shared_ptr<Tensor>> sigmoid_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs) {
    // Gradient of sigmoid: sigmoid(x) * (1 - sigmoid(x))
    auto& x = inputs[0];
    auto sig = std::make_shared<Tensor>(sigmoid(*x));
    auto one_minus_sig = std::make_shared<Tensor>(sub(Tensor::ones(sig->shape(), sig->device()), *sig));
    auto grad_sig = std::make_shared<Tensor>(mul(*sig, *one_minus_sig));
    
    // Use CUDA backward if available
    auto dx = std::make_shared<Tensor>(Tensor::zeros(sig->shape(), sig->device()));
    if (sig->device()->type() == DeviceType::CPU) {
        *dx = mul(*out_grad, *grad_sig);
    } else {
#ifdef ADAICPP_ENABLE_CUDA
        cuda::launch_sigmoid_backward(sig->data(), out_grad->data(), dx->data(), sig->size());
#else
        throw std::runtime_error("CUDA operations not enabled");
#endif
    }
    
    return {dx};
}

std::shared_ptr<Tensor> tanh_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("Tanh operation requires exactly 1 input");
    }
    return std::make_shared<Tensor>(tanh(*inputs[0]));
}

std::vector<std::shared_ptr<Tensor>> tanh_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs) {
    // Gradient of tanh: 1 - tanh(x)^2
    auto& x = inputs[0];
    auto t = std::make_shared<Tensor>(tanh(*x));
    auto t_squared = std::make_shared<Tensor>(mul(*t, *t));
    auto one_minus_t_squared = std::make_shared<Tensor>(sub(Tensor::ones(t_squared->shape(), t_squared->device()), *t_squared));
    
    // Use CUDA backward if available
    auto dx = std::make_shared<Tensor>(Tensor::zeros(t->shape(), t->device()));
    if (t->device()->type() == DeviceType::CPU) {
        *dx = mul(*out_grad, *one_minus_t_squared);
    } else {
#ifdef ADAICPP_ENABLE_CUDA
        cuda::launch_tanh_backward(t->data(), out_grad->data(), dx->data(), t->size());
#else
        throw std::runtime_error("CUDA operations not enabled");
#endif
    }
    
    return {dx};
}

std::shared_ptr<Tensor> pow_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 2) {
        throw std::runtime_error("Pow operation requires exactly 2 inputs");
    }
    return std::make_shared<Tensor>(pow(*inputs[0], *inputs[1]));
}

std::vector<std::shared_ptr<Tensor>> pow_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto& a = inputs[0];
    auto& b = inputs[1];
    
    auto b_minus_one = std::make_shared<Tensor>(sub(*b, Tensor::ones(b->shape(), b->device())));
    auto a_pow_b_minus_one = std::make_shared<Tensor>(pow(*a, *b_minus_one));
    auto da = std::make_shared<Tensor>(mul(mul(*out_grad, *b), *a_pow_b_minus_one));
    
    auto a_pow_b = std::make_shared<Tensor>(pow(*a, *b));
    auto log_a = std::make_shared<Tensor>(log(*a));
    auto db = std::make_shared<Tensor>(mul(mul(*out_grad, *a_pow_b), *log_a));
    
    return {da, db};
}

std::shared_ptr<Tensor> exp_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("Exp operation requires exactly 1 input");
    }
    return std::make_shared<Tensor>(exp(*inputs[0]));
}

std::vector<std::shared_ptr<Tensor>> exp_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto& x = inputs[0];
    auto exp_x = std::make_shared<Tensor>(exp(*x));
    auto dx = std::make_shared<Tensor>(mul(*out_grad, *exp_x));
    return {dx};
}

std::shared_ptr<Tensor> log_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("Log operation requires exactly 1 input");
    }
    return std::make_shared<Tensor>(log(*inputs[0]));
}

std::vector<std::shared_ptr<Tensor>> log_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto& x = inputs[0];
    auto ones = Tensor::ones(x->shape(), x->device());
    auto inv_x = std::make_shared<Tensor>(div(ones, *x));
    auto dx = std::make_shared<Tensor>(mul(*out_grad, *inv_x));
    return {dx};
}

std::shared_ptr<Tensor> softmax_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("Softmax operation requires exactly 1 input");
    }
    return std::make_shared<Tensor>(softmax(*inputs[0]));
}

std::vector<std::shared_ptr<Tensor>> softmax_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto& x = inputs[0];
    auto s = std::make_shared<Tensor>(softmax(*x));
    
    auto grad_times_softmax = std::make_shared<Tensor>(mul(*out_grad, *s));
    
    Tensor sum_grad_softmax({1}, x->device());
    if (x->device()->type() == DeviceType::CPU) {
        const float* gts_data = grad_times_softmax->data();
        float sum_val = 0.0f;
        for (size_t i = 0; i < grad_times_softmax->size(); ++i) {
            sum_val += gts_data[i];
        }
        sum_grad_softmax.data()[0] = sum_val;
    }
    
    auto grad_minus_sum = std::make_shared<Tensor>(sub(*out_grad, sum_grad_softmax));
    auto dx = std::make_shared<Tensor>(mul(*s, *grad_minus_sum));
    
    return {dx};
}

} // namespace adaicpp
