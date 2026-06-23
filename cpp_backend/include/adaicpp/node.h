#pragma once

#include "adaicpp/tensor.h"
#include "adaicpp/device.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace adaicpp {

// Forward declarations
class Graph;

enum class NodeType {
    Base,
    Placeholder,
    Parameter,
    Operation
};

class Node {
public:
    virtual ~Node() = default;
    
    // Accessors
    const std::string& name() const { return name_; }
    NodeType type() const { return type_; }
    bool is_static() const { return is_static_; }
    void set_is_static(bool is_static) { is_static_ = is_static; }
    std::shared_ptr<Device> device() const { return device_; }
    
    // Value and gradient
    std::shared_ptr<Tensor> value() { return value_; }
    const std::shared_ptr<Tensor> value() const { return value_; }
    std::shared_ptr<Tensor> grad() { return grad_; }
    const std::shared_ptr<Tensor> grad() const { return grad_; }
    void set_grad(std::shared_ptr<Tensor> grad) { grad_ = grad; }
    
    // Input and consumer management
    const std::vector<std::shared_ptr<Node>>& inputs() const { return inputs_; }
    std::vector<std::shared_ptr<Node>>& inputs() { return inputs_; }
    const std::vector<std::weak_ptr<Node>>& consumers() const { return consumers_; }
    std::vector<std::weak_ptr<Node>>& consumers() { return consumers_; }
    
    // Dirty flag for static nodes
    bool is_dirty() const { return dirty_; }
    void set_dirty(bool dirty) { dirty_ = dirty; }
    
    // Evaluation count for profiling
    size_t eval_count() const { return eval_count_; }
    
    // Virtual methods
    virtual void evaluate() = 0;
    virtual void backward(const std::shared_ptr<Tensor>& grad) = 0;
    
    // Dirty flag propagation
    virtual void mark_dirty();
    
    // Set value (for placeholder and parameter nodes)
    virtual void set_value(std::shared_ptr<Tensor> value);
    
protected:
    Node(const std::string& name, bool is_static, std::shared_ptr<Device> device);
    
    std::string name_;
    NodeType type_;
    bool is_static_;
    std::shared_ptr<Device> device_;
    
    std::shared_ptr<Tensor> value_;
    std::shared_ptr<Tensor> grad_;
    
    std::vector<std::shared_ptr<Node>> inputs_;
    std::vector<std::weak_ptr<Node>> consumers_;
    
    bool dirty_;
    size_t eval_count_;
    
    // Allow Graph and OpNode to access private members
    friend class Graph;
    friend class OpNode;
};

class PlaceholderNode : public Node {
public:
    PlaceholderNode(const std::string& name, std::shared_ptr<Device> device = nullptr);
    
    void evaluate() override;
    void backward(const std::shared_ptr<Tensor>& grad) override;
    
    void set_value(std::shared_ptr<Tensor> value) override;
};

class ParameterNode : public Node {
public:
    ParameterNode(const std::string& name, std::shared_ptr<Tensor> value);
    
    void evaluate() override;
    void backward(const std::shared_ptr<Tensor>& grad) override;
    
    void set_value(std::shared_ptr<Tensor> value) override;
};

using OpForwardFn = std::function<std::shared_ptr<Tensor>(const std::vector<std::shared_ptr<Tensor>>&)>;
using OpBackwardFn = std::function<std::vector<std::shared_ptr<Tensor>>(const std::shared_ptr<Tensor>&, const std::vector<std::shared_ptr<Tensor>>&)>;

class OpNode : public Node {
public:
    OpNode(
        const std::string& name,
        const std::vector<std::shared_ptr<Node>>& inputs,
        OpForwardFn forward_fn,
        OpBackwardFn backward_fn = nullptr,
        bool is_static = true,
        std::shared_ptr<Device> device = nullptr
    );
    
    void evaluate() override;
    void backward(const std::shared_ptr<Tensor>& grad) override;
    
    const std::string& op_name() const { return op_name_; }
    
    void set_op_name(const std::string& op_name) { op_name_ = op_name; }
    
    // Register this node as a consumer of its inputs (call after construction)
    void register_consumers(std::shared_ptr<Node> self);
    
private:
    std::string op_name_;
    OpForwardFn forward_fn_;
    OpBackwardFn backward_fn_;
    
    // Allow Graph to access private members
    friend class Graph;
};

} // namespace adaicpp
