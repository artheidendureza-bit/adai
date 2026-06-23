#pragma once

#include "adaicpp/graph.h"
#include "adaicpp/tensor.h"
#include <string>
#include <vector>
#include <fstream>

namespace adaicpp {

class ONNXExporter {
public:
    // Export a graph to ONNX format
    static std::string export_to_onnx(const std::shared_ptr<Graph>& graph,
                                     const std::string& model_name = "adai_model");
    
    // Save ONNX model to file
    static void save_onnx(const std::shared_ptr<Graph>& graph,
                        const std::string& filepath,
                        const std::string& model_name = "adai_model");
    
private:
    // Helper functions for ONNX serialization
    static std::string generate_onnx_proto(const std::shared_ptr<Graph>& graph,
                                          const std::string& model_name);
    
    static std::string tensor_to_onnx_value_info(const std::string& name,
                                                const std::vector<size_t>& shape,
                                                const std::string& dtype = "float");
    
    static std::string node_to_onnx_node(const std::shared_ptr<Node>& node);
    
    static std::string get_onnx_op_type(const std::string& op_name);
    
    static std::string shape_to_onnx_dim(const std::vector<size_t>& shape);
};

} // namespace adaicpp
