#include "adaicpp/node.h"
#include "adaicpp/device.h"
#include <stdexcept>
#include <algorithm>

namespace adaicpp {

// Node base class
Node::Node(const std::string& name, bool is_static, std::shared_ptr<Device> device)
    : name_(name), type_(NodeType::Base), is_static_(is_static),
      device_(device), dirty_(true), eval_count_(0) {
    
    if (!device_) {
        device_ = DeviceManager::instance().get_cpu_device();
    }
}

void Node::mark_dirty() {
    if (!dirty_) {
        dirty_ = true;
        for (auto& consumer_weak : consumers_) {
            if (auto consumer = consumer_weak.lock()) {
                consumer->mark_dirty();
            }
        }
    }
}

void Node::set_value(std::shared_ptr<Tensor> value) {
    value_ = value;
    mark_dirty();
    dirty_ = false;
}

// PlaceholderNode
PlaceholderNode::PlaceholderNode(const std::string& name, std::shared_ptr<Device> device)
    : Node(name, false, device) {
    type_ = NodeType::Placeholder;
}

void PlaceholderNode::evaluate() {
    if (!value_) {
        throw std::runtime_error("Placeholder " + name_ + " has no value set");
    }
}

void PlaceholderNode::backward(const std::shared_ptr<Tensor>& grad) {
    // Placeholders don't accumulate gradients
    // They are external inputs
}

void PlaceholderNode::set_value(std::shared_ptr<Tensor> value) {
    value_ = value;
    mark_dirty();
    dirty_ = false;
}

// ParameterNode
ParameterNode::ParameterNode(const std::string& name, std::shared_ptr<Tensor> value)
    : Node(name, true, value ? value->device() : nullptr) {
    type_ = NodeType::Parameter;
    value_ = value;
    dirty_ = false;  // Parameters start clean
}

void ParameterNode::evaluate() {
    // Parameters just return their stored value
    if (!value_) {
        throw std::runtime_error("Parameter " + name_ + " has no value");
    }
}

void ParameterNode::backward(const std::shared_ptr<Tensor>& grad) {
    // Accumulate gradient
    if (!grad_) {
        grad_ = std::make_shared<Tensor>(Tensor::zeros(grad->shape(), grad->device()));
    }
    
    // Add incoming gradient
    grad_ = std::make_shared<Tensor>(add(*grad_, *grad));
}

void ParameterNode::set_value(std::shared_ptr<Tensor> value) {
    value_ = value;
    mark_dirty();
    dirty_ = false;
}

// OpNode
OpNode::OpNode(
    const std::string& name,
    const std::vector<std::shared_ptr<Node>>& inputs,
    OpForwardFn forward_fn,
    OpBackwardFn backward_fn,
    bool is_static,
    std::shared_ptr<Device> device
) : Node(name, is_static, device), forward_fn_(forward_fn), backward_fn_(backward_fn) {
    type_ = NodeType::Operation;
    inputs_ = inputs;
    
    // Determine device from inputs if not specified
    if (!device_ && !inputs_.empty()) {
        device_ = inputs_[0]->device();
    }
    
    // By default, an operation is static if all its inputs are static
    if (is_static) {
        is_static_ = true;
    } else {
        is_static_ = std::all_of(inputs_.begin(), inputs_.end(),
            [](const auto& input) { return input->is_static(); });
    }
}

// Call this after the node is stored in a shared_ptr
void OpNode::register_consumers(std::shared_ptr<Node> self) {
    for (auto& input : inputs_) {
        input->consumers_.push_back(self);
    }
}

void OpNode::evaluate() {
    // If static and not dirty, return cached value
    if (is_static_ && !dirty_ && value_) {
        return;
    }
    
    // Evaluate inputs
    std::vector<std::shared_ptr<Tensor>> input_values;
    for (auto& input : inputs_) {
        input->evaluate();
        input_values.push_back(input->value());
    }
    
    // Apply forward function
    value_ = forward_fn_(input_values);
    eval_count_++;
    
    // Mark as clean if static
    if (is_static_) {
        dirty_ = false;
    }
}

void OpNode::backward(const std::shared_ptr<Tensor>& grad) {
    if (!backward_fn_) {
        // No gradient function defined
        return;
    }
    
    // Get input values
    std::vector<std::shared_ptr<Tensor>> input_values;
    for (auto& input : inputs_) {
        input_values.push_back(input->value());
    }
    
    // Compute gradients for inputs
    std::vector<std::shared_ptr<Tensor>> input_grads = backward_fn_(grad, input_values);
    
    // Propagate gradients to inputs
    for (size_t i = 0; i < inputs_.size() && i < input_grads.size(); ++i) {
        inputs_[i]->backward(input_grads[i]);
    }
}

} // namespace adaicpp
