#pragma once

#include "adaicpp/tensor.h"
#include "adaicpp/node.h"
#include <string>
#include <unordered_map>
#include <functional>
#include <memory>

namespace adaicpp {

class GraphOp {
public:
    GraphOp(
        const std::string& name,
        OpForwardFn forward_fn,
        OpBackwardFn backward_fn
    ) : name_(name), forward_fn_(forward_fn), backward_fn_(backward_fn) {}
    
    const std::string& name() const { return name_; }
    OpForwardFn forward_fn() const { return forward_fn_; }
    OpBackwardFn backward_fn() const { return backward_fn_; }
    
private:
    std::string name_;
    OpForwardFn forward_fn_;
    OpBackwardFn backward_fn_;
};

class OpRegistry {
public:
    static OpRegistry& instance();
    
    void register_op(const std::string& name, std::shared_ptr<GraphOp> op);
    std::shared_ptr<GraphOp> get_op(const std::string& name);
    bool has_op(const std::string& name) const;
    
    // Get forward function by name
    OpForwardFn get_forward_fn(const std::string& name);
    
    // Get backward function by name
    OpBackwardFn get_backward_fn(const std::string& name);
    
private:
    OpRegistry() = default;
    OpRegistry(const OpRegistry&) = delete;
    OpRegistry& operator=(const OpRegistry&) = delete;
    
    std::unordered_map<std::string, std::shared_ptr<GraphOp>> ops_;
};

// Convenience functions to register standard operations
void register_standard_operations();

// Forward/backward function declarations for standard operations
std::shared_ptr<Tensor> add_forward(const std::vector<std::shared_ptr<Tensor>>& inputs);
std::vector<std::shared_ptr<Tensor>> add_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs);

std::shared_ptr<Tensor> mul_forward(const std::vector<std::shared_ptr<Tensor>>& inputs);
std::vector<std::shared_ptr<Tensor>> mul_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs);

std::shared_ptr<Tensor> matmul_forward(const std::vector<std::shared_ptr<Tensor>>& inputs);
std::vector<std::shared_ptr<Tensor>> matmul_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs);

std::shared_ptr<Tensor> relu_forward(const std::vector<std::shared_ptr<Tensor>>& inputs);
std::vector<std::shared_ptr<Tensor>> relu_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs);

std::shared_ptr<Tensor> sigmoid_forward(const std::vector<std::shared_ptr<Tensor>>& inputs);
std::vector<std::shared_ptr<Tensor>> sigmoid_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs);

std::shared_ptr<Tensor> tanh_forward(const std::vector<std::shared_ptr<Tensor>>& inputs);
std::vector<std::shared_ptr<Tensor>> tanh_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs);

std::shared_ptr<Tensor> pow_forward(const std::vector<std::shared_ptr<Tensor>>& inputs);
std::vector<std::shared_ptr<Tensor>> pow_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs);

std::shared_ptr<Tensor> exp_forward(const std::vector<std::shared_ptr<Tensor>>& inputs);
std::vector<std::shared_ptr<Tensor>> exp_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs);

std::shared_ptr<Tensor> log_forward(const std::vector<std::shared_ptr<Tensor>>& inputs);
std::vector<std::shared_ptr<Tensor>> log_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs);

std::shared_ptr<Tensor> softmax_forward(const std::vector<std::shared_ptr<Tensor>>& inputs);
std::vector<std::shared_ptr<Tensor>> softmax_backward(const std::shared_ptr<Tensor>& out_grad, const std::vector<std::shared_ptr<Tensor>>& inputs);

} // namespace adaicpp
