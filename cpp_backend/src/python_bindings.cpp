#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "adaicpp/adaicpp.h"
#include "adaicpp/model.h"
#include "adaicpp/onnx_export.h"
#include "adaicpp/mixed_precision.h"

namespace py = pybind11;

// Helper function to convert numpy array to Tensor
std::shared_ptr<adaicpp::Tensor> numpy_to_tensor(py::array_t<float> array, std::shared_ptr<adaicpp::Device> device = nullptr) {
    py::buffer_info buf = array.request();
    
    if (buf.ndim == 0) {
        throw std::runtime_error("Cannot convert 0-dimensional array");
    }
    
    std::vector<size_t> shape(buf.ndim);
    for (size_t i = 0; i < buf.ndim; ++i) {
        shape[i] = buf.shape[i];
    }
    
    std::vector<float> data(static_cast<float*>(buf.ptr), static_cast<float*>(buf.ptr) + buf.size);
    return std::make_shared<adaicpp::Tensor>(shape, data, device);
}

// Helper function to convert Tensor to numpy array
py::array_t<float> tensor_to_numpy(const adaicpp::Tensor& tensor) {
    if (tensor.device()->type() == adaicpp::DeviceType::CUDA) {
        throw std::runtime_error("Cannot directly convert CUDA tensor to numpy. Use to_cpu() first.");
    }
    
    std::vector<size_t> shape = tensor.shape();
    std::vector<ssize_t> strides(shape.size());
    
    // Calculate strides for numpy
    for (size_t i = 0; i < shape.size(); ++i) {
        strides[i] = 1;
        for (size_t j = i + 1; j < shape.size(); ++j) {
            strides[i] *= shape[j];
        }
        strides[i] *= sizeof(float);
    }
    
    return py::array_t<float>(
        shape,
        strides,
        tensor.data()
    );
}

PYBIND11_MODULE(adaicpp, m) {
    m.doc() = "adAI C++ Backend Python Bindings";
    
    // Initialize the library
    m.def("initialize", &adaicpp::initialize, "Initialize the adAI C++ backend");
    
    // === Device ===
    py::enum_<adaicpp::DeviceType>(m, "DeviceType")
        .value("CPU", adaicpp::DeviceType::CPU)
        .value("CUDA", adaicpp::DeviceType::CUDA);
    
    py::class_<adaicpp::Device, std::shared_ptr<adaicpp::Device>>(m, "Device")
        .def("type", &adaicpp::Device::type)
        .def("device_id", &adaicpp::Device::device_id)
        .def("name", &adaicpp::Device::name);
    
    py::class_<adaicpp::CPUDevice, adaicpp::Device, std::shared_ptr<adaicpp::CPUDevice>>(m, "CPUDevice")
        .def(py::init<int>(), py::arg("device_id") = 0);
    
    // === DeviceManager ===
    py::class_<adaicpp::DeviceManager>(m, "DeviceManager")
        .def_static("instance", &adaicpp::DeviceManager::instance, py::return_value_policy::reference)
        .def("get_cpu_device", &adaicpp::DeviceManager::get_cpu_device, py::arg("device_id") = 0);
    
    // === Tensor ===
    py::class_<adaicpp::Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<size_t>&, std::shared_ptr<adaicpp::Device>>(), 
             py::arg("shape"), py::arg("device") = nullptr)
        .def(py::init<const std::vector<size_t>&, float, std::shared_ptr<adaicpp::Device>>(),
             py::arg("shape"), py::arg("value"), py::arg("device") = nullptr)
        .def("shape", &adaicpp::Tensor::shape)
        .def("ndim", &adaicpp::Tensor::ndim)
        .def("size", &adaicpp::Tensor::size)
        .def("data", py::overload_cast<>(&adaicpp::Tensor::data))
        .def("to_device", &adaicpp::Tensor::to_device)
        .def("to_cpu", &adaicpp::Tensor::to_cpu)
        .def("fill", &adaicpp::Tensor::fill)
        .def("zero", &adaicpp::Tensor::zero)
        .def("reshape", &adaicpp::Tensor::reshape)
        .def_static("zeros", &adaicpp::Tensor::zeros, py::arg("shape"), py::arg("device") = nullptr)
        .def_static("ones", &adaicpp::Tensor::ones, py::arg("shape"), py::arg("device") = nullptr)
        .def_static("randn", &adaicpp::Tensor::randn, py::arg("shape"), py::arg("device") = nullptr)
        .def_static("arange", &adaicpp::Tensor::arange, 
                    py::arg("start"), py::arg("stop"), py::arg("step") = 1.0f, py::arg("device") = nullptr);
    
    // === Tensor operations ===
    m.def("add", &adaicpp::add);
    m.def("sub", &adaicpp::sub);
    m.def("mul", &adaicpp::mul);
    m.def("div", &adaicpp::div);
    m.def("relu", &adaicpp::relu);
    m.def("sigmoid", &adaicpp::sigmoid);
    m.def("tanh", &adaicpp::tanh);
    m.def("matmul", &adaicpp::matmul);
    m.def("conv2d", &adaicpp::conv2d, py::arg("input"), py::arg("kernel"),
         py::arg("stride_h") = 1, py::arg("stride_w") = 1,
         py::arg("padding_h") = 0, py::arg("padding_w") = 0);
    m.def("sum", &adaicpp::sum);
    m.def("mean", &adaicpp::mean);
    
    // === MemoryPool ===
    py::class_<adaicpp::MemoryPool>(m, "MemoryPool")
        .def_static("instance", &adaicpp::MemoryPool::instance, py::return_value_policy::reference)
        .def("total_allocated_bytes", &adaicpp::MemoryPool::total_allocated_bytes)
        .def("total_cached_bytes", &adaicpp::MemoryPool::total_cached_bytes)
        .def("cache_hits", &adaicpp::MemoryPool::cache_hits)
        .def("cache_misses", &adaicpp::MemoryPool::cache_misses)
        .def("clear_all", &adaicpp::MemoryPool::clear_all);
    
    // === Node types ===
    py::enum_<adaicpp::NodeType>(m, "NodeType")
        .value("Base", adaicpp::NodeType::Base)
        .value("Placeholder", adaicpp::NodeType::Placeholder)
        .value("Parameter", adaicpp::NodeType::Parameter)
        .value("Operation", adaicpp::NodeType::Operation);
    
    py::class_<adaicpp::Node, std::shared_ptr<adaicpp::Node>>(m, "Node")
        .def("name", &adaicpp::Node::name)
        .def("type", &adaicpp::Node::type)
        .def("is_static", &adaicpp::Node::is_static)
        .def("set_is_static", &adaicpp::Node::set_is_static)
        .def("device", &adaicpp::Node::device)
        .def("value", py::overload_cast<>(&adaicpp::Node::value))
        .def("grad", py::overload_cast<>(&adaicpp::Node::grad))
        .def("inputs", py::overload_cast<>(&adaicpp::Node::inputs))
        .def("is_dirty", &adaicpp::Node::is_dirty)
        .def("set_dirty", &adaicpp::Node::set_dirty)
        .def("eval_count", &adaicpp::Node::eval_count);
    
    py::class_<adaicpp::PlaceholderNode, adaicpp::Node, std::shared_ptr<adaicpp::PlaceholderNode>>(m, "PlaceholderNode")
        .def(py::init<const std::string&, std::shared_ptr<adaicpp::Device>>(),
             py::arg("name"), py::arg("device") = nullptr)
        .def("set_value", &adaicpp::PlaceholderNode::set_value);
    
    py::class_<adaicpp::ParameterNode, adaicpp::Node, std::shared_ptr<adaicpp::ParameterNode>>(m, "ParameterNode")
        .def(py::init<const std::string&, std::shared_ptr<adaicpp::Tensor>>())
        .def("set_value", &adaicpp::ParameterNode::set_value);
    
    // Function types for operations
    using OpForwardFn = std::function<std::shared_ptr<adaicpp::Tensor>(const std::vector<std::shared_ptr<adaicpp::Tensor>>&)>;
    using OpBackwardFn = std::function<std::vector<std::shared_ptr<adaicpp::Tensor>>(const std::shared_ptr<adaicpp::Tensor>&, const std::vector<std::shared_ptr<adaicpp::Tensor>>&)>;
    
    py::class_<adaicpp::OpNode, adaicpp::Node, std::shared_ptr<adaicpp::OpNode>>(m, "OpNode")
        .def(py::init<const std::string&, const std::vector<std::shared_ptr<adaicpp::Node>>&, 
                      OpForwardFn, OpBackwardFn, bool, std::shared_ptr<adaicpp::Device>>(),
             py::arg("name"), py::arg("inputs"), py::arg("forward_fn"), py::arg("backward_fn") = nullptr,
             py::arg("is_static") = true, py::arg("device") = nullptr)
        .def("op_name", &adaicpp::OpNode::op_name)
        .def("set_op_name", &adaicpp::OpNode::set_op_name);
    
    // === Graph ===
    py::class_<adaicpp::Graph>(m, "Graph")
        .def(py::init<>())
        .def("add_node", &adaicpp::Graph::add_node)
        .def("remove_node", &adaicpp::Graph::remove_node)
        .def("get_node", &adaicpp::Graph::get_node)
        .def("has_node", &adaicpp::Graph::has_node)
        .def("add_placeholder", &adaicpp::Graph::add_placeholder,
             py::arg("name"), py::arg("device") = nullptr)
        .def("add_parameter", &adaicpp::Graph::add_parameter)
        .def("add_op", &adaicpp::Graph::add_op,
             py::arg("name"), py::arg("inputs"), py::arg("forward_fn"), 
             py::arg("backward_fn") = nullptr, py::arg("is_static") = true, py::arg("device") = nullptr)
        .def("set_inputs", &adaicpp::Graph::set_inputs)
        .def("modify_node_static", &adaicpp::Graph::modify_node_static)
        .def("compile", &adaicpp::Graph::compile)
        .def("forward", &adaicpp::Graph::forward)
        .def("backward", &adaicpp::Graph::backward)
        .def("optimize", &adaicpp::Graph::optimize)
        .def("sorted_nodes", &adaicpp::Graph::sorted_nodes)
        .def("is_compiled", &adaicpp::Graph::is_compiled)
        .def("get_value", &adaicpp::Graph::get_value)
        .def("get_grad", &adaicpp::Graph::get_grad)
        .def("zero_grad", &adaicpp::Graph::zero_grad)
        .def("to_json", &adaicpp::Graph::to_json)
        .def("save", &adaicpp::Graph::save)
        .def_static("load", &adaicpp::Graph::load)
        .def("to_mermaid", &adaicpp::Graph::to_mermaid)
        .def("print_layout", &adaicpp::Graph::print_layout);
    
    // === OpRegistry ===
    py::class_<adaicpp::GraphOp, std::shared_ptr<adaicpp::GraphOp>>(m, "GraphOp")
        .def(py::init<const std::string&, OpForwardFn, OpBackwardFn>())
        .def("name", &adaicpp::GraphOp::name)
        .def("forward_fn", &adaicpp::GraphOp::forward_fn)
        .def("backward_fn", &adaicpp::GraphOp::backward_fn);
    
    py::class_<adaicpp::OpRegistry>(m, "OpRegistry")
        .def_static("instance", &adaicpp::OpRegistry::instance, py::return_value_policy::reference)
        .def("register_op", &adaicpp::OpRegistry::register_op)
        .def("get_op", &adaicpp::OpRegistry::get_op)
        .def("has_op", &adaicpp::OpRegistry::has_op)
        .def("get_forward_fn", &adaicpp::OpRegistry::get_forward_fn)
        .def("get_backward_fn", &adaicpp::OpRegistry::get_backward_fn);
    
    m.def("register_standard_operations", &adaicpp::register_standard_operations);
    
    // Standard operation forward/backward functions
    m.def("add_forward", &adaicpp::add_forward);
    m.def("add_backward", &adaicpp::add_backward);
    m.def("mul_forward", &adaicpp::mul_forward);
    m.def("mul_backward", &adaicpp::mul_backward);
    m.def("matmul_forward", &adaicpp::matmul_forward);
    m.def("matmul_backward", &adaicpp::matmul_backward);
    m.def("relu_forward", &adaicpp::relu_forward);
    m.def("relu_backward", &adaicpp::relu_backward);
    m.def("sigmoid_forward", &adaicpp::sigmoid_forward);
    m.def("sigmoid_backward", &adaicpp::sigmoid_backward);
    m.def("tanh_forward", &adaicpp::tanh_forward);
    m.def("tanh_backward", &adaicpp::tanh_backward);
    
    // === Model classes ===
    py::class_<adaicpp::Layer, std::shared_ptr<adaicpp::Layer>>(m, "Layer")
        .def("name", &adaicpp::Layer::name)
        .def("parameters", &adaicpp::Layer::parameters)
        .def("set_parameters", &adaicpp::Layer::set_parameters);
    
    py::class_<adaicpp::DenseLayer, adaicpp::Layer, std::shared_ptr<adaicpp::DenseLayer>>(m, "DenseLayer")
        .def(py::init<int, int, const std::string&, const std::string>(),
             py::arg("input_size"), py::arg("output_size"), py::arg("activation") = "relu", py::arg("name") = "dense");
    
    py::class_<adaicpp::Conv2DLayer, adaicpp::Layer, std::shared_ptr<adaicpp::Conv2DLayer>>(m, "Conv2DLayer")
        .def(py::init<int, int, int, int, int, const std::string&, const std::string>(),
             py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"),
             py::arg("stride") = 1, py::arg("padding") = 0,
             py::arg("activation") = "relu", py::arg("name") = "conv2d");
    
    py::class_<adaicpp::TrainingCallback, std::shared_ptr<adaicpp::TrainingCallback>>(m, "TrainingCallback")
        .def(py::init<>())
        .def("on_epoch_begin", &adaicpp::TrainingCallback::on_epoch_begin)
        .def("on_epoch_end", &adaicpp::TrainingCallback::on_epoch_end)
        .def("on_batch_begin", &adaicpp::TrainingCallback::on_batch_begin)
        .def("on_batch_end", &adaicpp::TrainingCallback::on_batch_end);
    
    py::class_<adaicpp::Model, std::shared_ptr<adaicpp::Model>>(m, "Model")
        .def(py::init<const std::string>(), py::arg("name") = "model")
        .def("add_layer", &adaicpp::Model::add_layer)
        .def("build", &adaicpp::Model::build)
        .def("compile", &adaicpp::Model::compile,
             py::arg("optimizer") = "adam", py::arg("learning_rate") = 0.001f, py::arg("loss") = "mse")
        .def("fit", &adaicpp::Model::fit,
             py::arg("x_train"), py::arg("y_train"),
             py::arg("epochs") = 10, py::arg("batch_size") = 32,
             py::arg("x_val") = py::none(), py::arg("y_val") = py::none(),
             py::arg("callbacks") = py::none())
        .def("predict", &adaicpp::Model::predict)
        .def("evaluate", &adaicpp::Model::evaluate)
        .def("save", &adaicpp::Model::save)
        .def_static("load", &adaicpp::Model::load)
        .def("graph", &adaicpp::Model::graph)
        .def("name", &adaicpp::Model::name);
    
    py::class_<adaicpp::Sequential, adaicpp::Model, std::shared_ptr<adaicpp::Sequential>>(m, "Sequential")
        .def(py::init<const std::string>(), py::arg("name") = "sequential");
    
    py::class_<adaicpp::MLP, adaicpp::Model, std::shared_ptr<adaicpp::MLP>>(m, "MLP")
        .def(py::init<const std::vector<int>&, const std::string&, const std::string>(),
             py::arg("hidden_sizes"), py::arg("activation") = "relu", py::arg("name") = "mlp");
    
    py::class_<adaicpp::CNN, adaicpp::Model, std::shared_ptr<adaicpp::CNN>>(m, "CNN")
        .def(py::init<const std::vector<int>&, const std::vector<int>&, const std::string&, const std::string>(),
             py::arg("conv_channels"), py::arg("dense_sizes"),
             py::arg("activation") = "relu", py::arg("name") = "cnn");
    
    // === ONNX Export ===
    py::class_<adaicpp::ONNXExporter>(m, "ONNXExporter")
        .def_static("export_to_onnx", &adaicpp::ONNXExporter::export_to_onnx,
             py::arg("graph"), py::arg("model_name") = "adai_model")
        .def_static("save_onnx", &adaicpp::ONNXExporter::save_onnx,
             py::arg("graph"), py::arg("filepath"), py::arg("model_name") = "adai_model");
    
    // === Mixed Precision ===
    py::enum_<adaicpp::Precision>(m, "Precision")
        .value("FP32", adaicpp::Precision::FP32)
        .value("FP16", adaicpp::Precision::FP16)
        .value("MIXED", adaicpp::Precision::MIXED);
    
    py::class_<adaicpp::MixedPrecisionTrainer>(m, "MixedPrecisionTrainer")
        .def(py::init<std::shared_ptr<adaicpp::Model>, float>(),
             py::arg("model"), py::arg("loss_scale") = 1.0f)
        .def("convert_to_mixed_precision", &adaicpp::MixedPrecisionTrainer::convert_to_mixed_precision)
        .def("training_step", &adaicpp::MixedPrecisionTrainer::training_step)
        .def("get_loss_scale", &adaicpp::MixedPrecisionTrainer::get_loss_scale)
        .def("update_loss_scale", &adaicpp::MixedPrecisionTrainer::update_loss_scale);
    
    m.def("convert_precision", &adaicpp::convert_precision);
    m.def("supports_mixed_precision", &adaicpp::supports_mixed_precision);
    
    // === Helper functions for numpy interop ===
    m.def("numpy_to_tensor", &numpy_to_tensor, py::arg("array"), py::arg("device") = nullptr);
    m.def("tensor_to_numpy", &tensor_to_numpy);
}
