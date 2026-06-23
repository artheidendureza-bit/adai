#include "adaicpp/graph.h"
#include "adaicpp/node.h"
#include "adaicpp/op_registry.h"
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_set>
#include <functional>

namespace adaicpp {

Graph::Graph() : needs_sorting_(true), compiled_(false) {}

std::shared_ptr<Node> Graph::add_node(std::shared_ptr<Node> node) {
    if (nodes_.find(node->name()) != nodes_.end()) {
        throw std::runtime_error("Node with name " + node->name() + " already exists in graph");
    }
    
    nodes_[node->name()] = node;
    needs_sorting_ = true;
    compiled_ = false;
    
    return node;
}

void Graph::remove_node(const std::string& name) {
    auto it = nodes_.find(name);
    if (it == nodes_.end()) {
        throw std::runtime_error("Node " + name + " does not exist in graph");
    }
    
    auto node = it->second;
    
    // Disconnect from inputs
    for (auto& input : node->inputs()) {
        auto& consumers = input->consumers();
        consumers.erase(
            std::remove_if(consumers.begin(), consumers.end(),
                [&node](const std::weak_ptr<Node>& w) {
                    return w.lock() == node;
                }),
            consumers.end()
        );
    }
    
    // Disconnect consumers
    for (auto& consumer_weak : node->consumers()) {
        if (auto consumer = consumer_weak.lock()) {
            auto& inputs = consumer->inputs();
            inputs.erase(
                std::remove(inputs.begin(), inputs.end(), node),
                inputs.end()
            );
            consumer->mark_dirty();
        }
    }
    
    nodes_.erase(it);
    needs_sorting_ = true;
    compiled_ = false;
}

std::shared_ptr<Node> Graph::get_node(const std::string& name) {
    auto it = nodes_.find(name);
    if (it == nodes_.end()) {
        throw std::runtime_error("Node " + name + " does not exist in graph");
    }
    return it->second;
}

bool Graph::has_node(const std::string& name) const {
    return nodes_.find(name) != nodes_.end();
}

std::shared_ptr<PlaceholderNode> Graph::add_placeholder(const std::string& name, std::shared_ptr<Device> device) {
    auto node = std::make_shared<PlaceholderNode>(name, device);
    add_node(node);
    return node;
}

std::shared_ptr<ParameterNode> Graph::add_parameter(const std::string& name, std::shared_ptr<Tensor> value) {
    auto node = std::make_shared<ParameterNode>(name, value);
    add_node(node);
    return node;
}

std::shared_ptr<OpNode> Graph::add_op(
    const std::string& name,
    const std::vector<std::shared_ptr<Node>>& inputs,
    OpForwardFn forward_fn,
    OpBackwardFn backward_fn,
    bool is_static,
    std::shared_ptr<Device> device
) {
    auto node = std::make_shared<OpNode>(name, inputs, forward_fn, backward_fn, is_static, device);
    add_node(node);
    // Register consumers after the node is in a shared_ptr
    node->register_consumers(node);
    return node;
}

std::shared_ptr<OpNode> Graph::add_op(
    const std::string& name,
    const std::string& op_name,
    const std::vector<std::shared_ptr<Node>>& inputs,
    std::shared_ptr<Device> device
) {
    auto& registry = OpRegistry::instance();
    auto forward_fn = registry.get_forward_fn(op_name);
    auto backward_fn = registry.get_backward_fn(op_name);
    return add_op(name, inputs, forward_fn, backward_fn, true, device);
}

void Graph::set_inputs(const std::string& name, const std::vector<std::shared_ptr<Node>>& new_inputs) {
    auto node = get_node(name);
    
    // Disconnect from old inputs
    for (auto& input : node->inputs()) {
        auto& consumers = input->consumers();
        consumers.erase(
            std::remove_if(consumers.begin(), consumers.end(),
                [&node](const std::weak_ptr<Node>& w) {
                    return w.lock() == node;
                }),
            consumers.end()
        );
    }
    
    // Set new inputs
    node->inputs() = new_inputs;
    
    // Connect to new inputs
    for (auto& input : new_inputs) {
        input->consumers().push_back(node);
    }
    
    node->mark_dirty();
    needs_sorting_ = true;
    compiled_ = false;
}

void Graph::modify_node_static(const std::string& name, bool is_static) {
    auto node = get_node(name);
    if (node->is_static() != is_static) {
        node->set_is_static(is_static);
        node->mark_dirty();
        compiled_ = false;
    }
}

void Graph::compile() {
    if (compiled_ && !needs_sorting_) {
        return;
    }
    
    // Topological sort
    if (needs_sorting_) {
        topological_sort();
        needs_sorting_ = false;
    }
    
    // Build compiled instructions
    compiled_instructions_.clear();
    for (auto& node : sorted_nodes_) {
        if (auto op_node = std::dynamic_pointer_cast<OpNode>(node)) {
            compiled_instructions_.push_back({
                op_node,
                op_node->forward_fn_,  // Access private member (friend class)
                op_node->inputs(),
                op_node->is_static()
            });
        }
    }
    
    compiled_ = true;
}

void Graph::forward(const std::unordered_map<std::string, std::shared_ptr<Tensor>>& feeds) {
    compile();
    
    // Update placeholder values
    for (const auto& [name, value] : feeds) {
        auto node = get_node(name);
        auto placeholder = std::dynamic_pointer_cast<PlaceholderNode>(node);
        if (!placeholder) {
            throw std::runtime_error("Node " + name + " is not a PlaceholderNode");
        }
        placeholder->set_value(value);
    }
    
    // Execute compiled instructions
    for (auto& instr : compiled_instructions_) {
        auto node = instr.node;
        
        // Skip static nodes that are not dirty
        if (instr.is_static && !node->is_dirty() && node->value()) {
            continue;
        }
        
        // Evaluate inputs
        std::vector<std::shared_ptr<Tensor>> input_values;
        for (auto& input : instr.inputs) {
            input_values.push_back(input->value());
        }
        
        // Apply operation
        node->value() = instr.op(input_values);
        
        // Mark as clean if static
        if (instr.is_static) {
            node->set_dirty(false);
        }
    }
}

void Graph::backward(const std::string& target_node_name) {
    compile();
    
    auto target = get_node(target_node_name);
    if (!target->value()) {
        throw std::runtime_error("Cannot run backward on " + target_node_name + " as it has no computed value");
    }
    
    // Reset all gradients
    zero_grad();
    
    // Seed target gradient with ones
    auto ones = Tensor::ones(target->value()->shape(), target->value()->device());
    target->grad() = std::make_shared<Tensor>(ones);
    
    // Walk backward through topologically sorted nodes
    for (auto it = sorted_nodes_.rbegin(); it != sorted_nodes_.rend(); ++it) {
        auto node = *it;
        
        if (!node->grad()) {
            continue;
        }
        
        auto op_node = std::dynamic_pointer_cast<OpNode>(node);
        if (!op_node) {
            continue;
        }
        
        // Backward pass is handled by the node itself
        op_node->backward(node->grad());
    }
}

void Graph::optimize(const std::vector<std::string>& outputs) {
    // Dead code elimination
    dead_code_elimination(outputs);
    
    // Re-compile
    needs_sorting_ = true;
    compile();
    
    // Constant folding
    constant_folding();
    
    compile();
}

std::shared_ptr<Tensor> Graph::get_value(const std::string& name) {
    auto node = get_node(name);
    return node->value();
}

std::shared_ptr<Tensor> Graph::get_grad(const std::string& name) {
    auto node = get_node(name);
    return node->grad();
}

void Graph::zero_grad() {
    for (auto& [name, node] : nodes_) {
        node->grad() = nullptr;
    }
}

std::string Graph::to_json() const {
    std::ostringstream oss;
    oss << "{\n  \"nodes\": [\n";
    
    bool first = true;
    for (const auto& [name, node] : nodes_) {
        if (!first) {
            oss << ",\n";
        }
        first = false;
        
        oss << "    {\n";
        oss << "      \"name\": \"" << node->name() << "\",\n";
        oss << "      \"type\": ";
        
        if (std::dynamic_pointer_cast<PlaceholderNode>(node)) {
            oss << "\"PlaceholderNode\"";
        } else if (std::dynamic_pointer_cast<ParameterNode>(node)) {
            oss << "\"ParameterNode\"";
        } else if (std::dynamic_pointer_cast<OpNode>(node)) {
            oss << "\"OpNode\"";
        } else {
            oss << "\"Node\"";
        }
        oss << ",\n";
        
        oss << "      \"is_static\": " << (node->is_static() ? "true" : "false") << ",\n";
        
        oss << "      \"inputs\": [";
        bool first_input = true;
        for (const auto& input : node->inputs()) {
            if (!first_input) {
                oss << ", ";
            }
            first_input = false;
            oss << "\"" << input->name() << "\"";
        }
        oss << "]\n";
        
        oss << "    }";
    }
    
    oss << "\n  ]\n}";
    
    return oss.str();
}

std::unique_ptr<Graph> Graph::from_json(const std::string& json_str) {
    // This is a simplified implementation
    // A full implementation would use a JSON library
    auto graph = std::make_unique<Graph>();
    
    // Parse JSON and reconstruct nodes
    // This is complex and would require a proper JSON parser
    // For now, we'll return an empty graph
    
    return graph;
}

void Graph::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    file << to_json();
}

std::unique_ptr<Graph> Graph::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    
    std::string json_str((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    
    return from_json(json_str);
}

std::string Graph::to_mermaid() const {
    const_cast<Graph*>(this)->compile();
    
    std::ostringstream oss;
    oss << "graph TD\n";
    
    for (const auto& node : sorted_nodes_) {
        std::string node_style;
        std::string node_label;
        
        if (auto placeholder = std::dynamic_pointer_cast<PlaceholderNode>(node)) {
            node_style = "fill:#e1f5fe,stroke:#0288d1,stroke-width:2px";
            node_label = node->name() + " [Placeholder]";
        } else if (auto parameter = std::dynamic_pointer_cast<ParameterNode>(node)) {
            node_style = "fill:#efebe9,stroke:#5d4037,stroke-width:2px";
            node_label = node->name() + " [Parameter" + (node->is_static() ? " (Static)" : " (Dynamic)") + "]";
        } else if (auto op_node = std::dynamic_pointer_cast<OpNode>(node)) {
            node_style = "fill:#e8f5e9,stroke:#388e3c,stroke-width:2px";
            node_label = node->name() + " [Op: " + op_node->op_name() + (node->is_static() ? " (Static)" : " (Dynamic)") + "]";
        } else {
            node_label = node->name();
        }
        
        oss << "    " << node->name() << "[\"" << node_label << "\"]\n";
        oss << "    style " << node->name() << " " << node_style << "\n";
        
        for (const auto& consumer_weak : node->consumers()) {
            if (auto consumer = consumer_weak.lock()) {
                oss << "    " << node->name() << " --> " << consumer->name() << "\n";
            }
        }
    }
    
    return oss.str();
}

void Graph::print_layout() const {
    const_cast<Graph*>(this)->compile();
    
    std::cout << "\n=== Graph Compilation Layout ===" << std::endl;
    for (size_t i = 0; i < sorted_nodes_.size(); ++i) {
        const auto& node = sorted_nodes_[i];
        
        std::string inputs_str;
        if (!node->inputs().empty()) {
            for (size_t j = 0; j < node->inputs().size(); ++j) {
                if (j > 0) inputs_str += ", ";
                inputs_str += node->inputs()[j]->name();
            }
        } else {
            inputs_str = "None";
        }
        
        std::string node_type;
        if (std::dynamic_pointer_cast<PlaceholderNode>(node)) {
            node_type = "PlaceholderNode";
        } else if (std::dynamic_pointer_cast<ParameterNode>(node)) {
            node_type = "ParameterNode";
        } else if (std::dynamic_pointer_cast<OpNode>(node)) {
            node_type = "OpNode";
        } else {
            node_type = "Node";
        }
        
        std::cout << "[" << i << "] Node: " << node->name() 
                  << " | Type: " << node_type
                  << " | Inputs: [" << inputs_str << "]"
                  << " | Device: " << (node->device() ? node->device()->name() : "None")
                  << std::endl;
    }
    std::cout << "================================" << std::endl;
}

void Graph::topological_sort() {
    sorted_nodes_.clear();
    
    std::unordered_map<std::string, int> visited;
    for (const auto& [name, node] : nodes_) {
        visited[name] = 0;  // 0 = unvisited, 1 = visiting, 2 = visited
    }
    
    for (const auto& [name, node] : nodes_) {
        if (visited[name] == 0) {
            dfs_visit(node, visited, sorted_nodes_);
        }
    }
    
    // Reverse to get correct topological order
    std::reverse(sorted_nodes_.begin(), sorted_nodes_.end());
}

void Graph::dfs_visit(std::shared_ptr<Node> node, std::unordered_map<std::string, int>& visited, std::vector<std::shared_ptr<Node>>& order) {
    visited[node->name()] = 1;  // visiting
    
    for (const auto& input : node->inputs()) {
        if (visited[input->name()] == 1) {
            throw std::runtime_error("Cycle detected in computational graph!");
        }
        if (visited[input->name()] == 0) {
            dfs_visit(input, visited, order);
        }
    }
    
    visited[node->name()] = 2;  // visited
    order.push_back(node);
}

void Graph::dead_code_elimination(const std::vector<std::string>& outputs) {
    std::unordered_set<std::shared_ptr<Node>> reachable;
    
    std::function<void(std::shared_ptr<Node>)> dfs = [&](std::shared_ptr<Node> node) {
        if (reachable.find(node) != reachable.end()) {
            return;
        }
        reachable.insert(node);
        for (const auto& input : node->inputs()) {
            dfs(input);
        }
    };
    
    // Mark all nodes reachable from outputs
    for (const auto& output_name : outputs) {
        dfs(get_node(output_name));
    }
    
    // Remove unreachable nodes
    std::vector<std::string> to_remove;
    for (const auto& [name, node] : nodes_) {
        if (reachable.find(node) == reachable.end()) {
            to_remove.push_back(name);
        }
    }
    
    for (const auto& name : to_remove) {
        remove_node(name);
    }
}

void Graph::constant_folding() {
    bool folded_any = true;
    
    while (folded_any) {
        folded_any = false;
        compile();
        
        for (auto& node : sorted_nodes_) {
            auto op_node = std::dynamic_pointer_cast<OpNode>(node);
            if (!op_node || !op_node->is_static()) {
                continue;
            }
            
            // Check if all inputs are static
            bool all_static = true;
            for (const auto& input : op_node->inputs()) {
                if (!input->is_static()) {
                    all_static = false;
                    break;
                }
            }
            
            if (!all_static) {
                continue;
            }
            
            // Evaluate the static value
            op_node->evaluate();
            auto folded_value = op_node->value();
            
            // Replace OpNode with ParameterNode
            auto new_node = std::make_shared<ParameterNode>(op_node->name(), folded_value);
            
            // Update consumers
            for (auto& consumer_weak : op_node->consumers()) {
                if (auto consumer = consumer_weak.lock()) {
                    auto& inputs = consumer->inputs();
                    for (auto& input : inputs) {
                        if (input == op_node) {
                            input = new_node;
                        }
                    }
                    new_node->consumers().push_back(consumer);
                }
            }
            
            // Remove old node and add new one
            nodes_[op_node->name()] = new_node;
            
            folded_any = true;
            needs_sorting_ = true;
            compiled_ = false;
            break;  // Break to re-sort and repeat
        }
    }
}

} // namespace adaicpp
