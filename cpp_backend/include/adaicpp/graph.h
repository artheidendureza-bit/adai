#pragma once

#include "adaicpp/node.h"
#include "adaicpp/tensor.h"
#include "adaicpp/device.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

namespace adaicpp {

class Graph {
public:
    Graph();
    ~Graph() = default;
    
    // Node management
    std::shared_ptr<Node> add_node(std::shared_ptr<Node> node);
    void remove_node(const std::string& name);
    std::shared_ptr<Node> get_node(const std::string& name);
    bool has_node(const std::string& name) const;
    
    // Convenience methods for creating nodes
    std::shared_ptr<PlaceholderNode> add_placeholder(const std::string& name, std::shared_ptr<Device> device = nullptr);
    std::shared_ptr<ParameterNode> add_parameter(const std::string& name, std::shared_ptr<Tensor> value);
    std::shared_ptr<OpNode> add_op(
        const std::string& name,
        const std::vector<std::shared_ptr<Node>>& inputs,
        OpForwardFn forward_fn,
        OpBackwardFn backward_fn = nullptr,
        bool is_static = true,
        std::shared_ptr<Device> device = nullptr
    );
    std::shared_ptr<OpNode> add_op(
        const std::string& name,
        const std::string& op_name,
        const std::vector<std::shared_ptr<Node>>& inputs,
        std::shared_ptr<Device> device = nullptr
    );
    
    // Graph modification
    void set_inputs(const std::string& name, const std::vector<std::shared_ptr<Node>>& new_inputs);
    void modify_node_static(const std::string& name, bool is_static);
    
    // Compilation and execution
    void compile();
    void forward(const std::unordered_map<std::string, std::shared_ptr<Tensor>>& feeds);
    void backward(const std::string& target_node_name);
    
    // Graph optimization
    void optimize(const std::vector<std::string>& outputs);
    
    // Accessors
    const std::unordered_map<std::string, std::shared_ptr<Node>>& nodes() const { return nodes_; }
    const std::vector<std::shared_ptr<Node>>& sorted_nodes() const { return sorted_nodes_; }
    bool is_compiled() const { return compiled_; }
    
    // Get node values
    std::shared_ptr<Tensor> get_value(const std::string& name);
    std::shared_ptr<Tensor> get_grad(const std::string& name);
    
    // Clear all gradients
    void zero_grad();
    
    // Serialization
    std::string to_json() const;
    static std::unique_ptr<Graph> from_json(const std::string& json_str);
    void save(const std::string& path) const;
    static std::unique_ptr<Graph> load(const std::string& path);
    
    // Visualization
    std::string to_mermaid() const;
    void print_layout() const;
    
private:
    void topological_sort();
    void dfs_visit(std::shared_ptr<Node> node, std::unordered_map<std::string, int>& visited, std::vector<std::shared_ptr<Node>>& order);
    
    void dead_code_elimination(const std::vector<std::string>& outputs);
    void constant_folding();
    
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes_;
    std::vector<std::shared_ptr<Node>> sorted_nodes_;
    bool needs_sorting_;
    bool compiled_;
    
    struct CompiledInstruction {
        std::shared_ptr<OpNode> node;
        OpForwardFn op;
        std::vector<std::shared_ptr<Node>> inputs;
        bool is_static;
    };
    std::vector<CompiledInstruction> compiled_instructions_;
};

} // namespace adaicpp
