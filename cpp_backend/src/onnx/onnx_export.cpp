#include "adaicpp/onnx_export.h"
#include "adaicpp/graph.h"
#include "adaicpp/node.h"
#include <sstream>
#include <algorithm>
#include <fstream>

namespace adaicpp {

std::string ONNXExporter::export_to_onnx(const std::shared_ptr<Graph>& graph,
                                        const std::string& model_name) {
    return generate_onnx_proto(graph, model_name);
}

void ONNXExporter::save_onnx(const std::shared_ptr<Graph>& graph,
                            const std::string& filepath,
                            const std::string& model_name) {
    std::string onnx_proto = export_to_onnx(graph, model_name);
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    file << onnx_proto;
}

std::string ONNXExporter::generate_onnx_proto(const std::shared_ptr<Graph>& graph,
                                              const std::string& model_name) {
    std::ostringstream oss;
    
    // ONNX IR format (simplified text representation)
    oss << "ir_version: 7\n";
    oss << "producer_name: \"adAI\"\n";
    oss << "producer_version: \"0.1.0\"\n";
    oss << "graph {\n";
    oss << "  name: \"" << model_name << "\"\n";
    
    // Get input nodes (placeholders)
    std::vector<std::shared_ptr<Node>> input_nodes;
    std::vector<std::shared_ptr<Node>> output_nodes;
    
    for (const auto& [name, node] : graph->nodes()) {
        if (node->type() == NodeType::Placeholder) {
            input_nodes.push_back(node);
        }
        // Assume last node is output for simplicity
        if (node->consumers().empty()) {
            output_nodes.push_back(node);
        }
    }
    
    // Add inputs
    oss << "  input {\n";
    for (const auto& node : input_nodes) {
        oss << "    name: \"" << node->name() << "\"\n";
        // Use default shape for placeholders
        oss << "    type {\n";
        oss << "      tensor_type {\n";
        oss << "        elem_type: 1\n";  // FLOAT
        oss << "        shape {\n";
        oss << "          dim {\n";
        oss << "            dim_param: \"batch\"\n";
        oss << "          }\n";
        oss << "        }\n";
        oss << "      }\n";
        oss << "    }\n";
    }
    oss << "  }\n";
    
    // Add outputs
    oss << "  output {\n";
    for (const auto& node : output_nodes) {
        oss << "    name: \"" << node->name() << "\"\n";
        oss << "    type {\n";
        oss << "      tensor_type {\n";
        oss << "        elem_type: 1\n";  // FLOAT
        oss << "        shape {\n";
        oss << "          dim {\n";
        oss << "            dim_param: \"batch\"\n";
        oss << "          }\n";
        oss << "        }\n";
        oss << "      }\n";
        oss << "    }\n";
    }
    oss << "  }\n";
    
    // Add nodes
    oss << "  node {\n";
    for (const auto& [name, node] : graph->nodes()) {
        if (node->type() == NodeType::Operation) {
            auto op_node = std::dynamic_pointer_cast<OpNode>(node);
            if (op_node) {
                oss << "    {\n";
                oss << "      name: \"" << node->name() << "\"\n";
                oss << "      op_type: \"" << get_onnx_op_type(op_node->op_name()) << "\"\n";
                oss << "      input: [";
                for (size_t i = 0; i < node->inputs().size(); i++) {
                    if (i > 0) oss << ", ";
                    oss << "\"" << node->inputs()[i]->name() << "\"";
                }
                oss << "]\n";
                oss << "      output: [\"" << node->name() << "\"]\n";
                oss << "    }\n";
            }
        }
    }
    oss << "  }\n";
    
    // Add initializer (parameters)
    oss << "  initializer {\n";
    for (const auto& [name, node] : graph->nodes()) {
        if (node->type() == NodeType::Parameter) {
            auto param_node = std::dynamic_pointer_cast<ParameterNode>(node);
            if (param_node && param_node->value()) {
                oss << "    {\n";
                oss << "      name: \"" << node->name() << "\"\n";
                oss << "      dims: " << param_node->value()->shape().size() << "\n";
                for (size_t dim : param_node->value()->shape()) {
                    oss << "      dims: " << dim << "\n";
                }
                oss << "      data_type: 1\n";  // FLOAT
                // Note: In a full implementation, we would serialize the actual tensor data
                oss << "    }\n";
            }
        }
    }
    oss << "  }\n";
    
    oss << "}\n";
    
    return oss.str();
}

std::string ONNXExporter::tensor_to_onnx_value_info(const std::string& name,
                                                    const std::vector<size_t>& shape,
                                                    const std::string& dtype) {
    (void)dtype;  // Unused in current implementation
    std::ostringstream oss;
    oss << "  {\n";
    oss << "    name: \"" << name << "\"\n";
    oss << "    type {\n";
    oss << "      tensor_type {\n";
    oss << "        elem_type: 1\n";  // FLOAT
    oss << "        shape {\n";
    for (size_t dim : shape) {
        oss << "          dim {\n";
        oss << "            dim_value: " << dim << "\n";
        oss << "          }\n";
    }
    oss << "        }\n";
    oss << "      }\n";
    oss << "    }\n";
    oss << "  }\n";
    return oss.str();
}

std::string ONNXExporter::node_to_onnx_node(const std::shared_ptr<Node>& node) {
    std::ostringstream oss;
    auto op_node = std::dynamic_pointer_cast<OpNode>(node);
    if (!op_node) {
        return "";
    }
    
    oss << "  {\n";
    oss << "    name: \"" << node->name() << "\"\n";
    oss << "    op_type: \"" << get_onnx_op_type(op_node->op_name()) << "\"\n";
    oss << "    input: [";
    for (size_t i = 0; i < node->inputs().size(); i++) {
        if (i > 0) oss << ", ";
        oss << "\"" << node->inputs()[i]->name() << "\"";
    }
    oss << "]\n";
    oss << "    output: [\"" << node->name() << "\"]\n";
    oss << "  }\n";
    
    return oss.str();
}

std::string ONNXExporter::get_onnx_op_type(const std::string& op_name) {
    // Map adAI operation names to ONNX operation names
    if (op_name == "add") return "Add";
    if (op_name == "sub") return "Sub";
    if (op_name == "mul") return "Mul";
    if (op_name == "div") return "Div";
    if (op_name == "matmul") return "MatMul";
    if (op_name == "relu") return "Relu";
    if (op_name == "sigmoid") return "Sigmoid";
    if (op_name == "tanh") return "Tanh";
    if (op_name == "conv2d") return "Conv";
    if (op_name == "maxpool2d") return "MaxPool";
    
    // Default: return as-is
    return op_name;
}

std::string ONNXExporter::shape_to_onnx_dim(const std::vector<size_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

} // namespace adaicpp
