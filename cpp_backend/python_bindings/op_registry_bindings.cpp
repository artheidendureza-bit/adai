#include "adaicpp/op_registry.h"
#include "adaicpp/tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace adaicpp {

void bind_op_registry(py::module& m) {
    // GraphOp class
    py::class_<GraphOp, std::shared_ptr<GraphOp>>(m, "GraphOp")
        .def_property_readonly("name", &GraphOp::name)
        .def_property_readonly("forward_fn", &GraphOp::forward_fn)
        .def_property_readonly("backward_fn", &GraphOp::backward_fn);
    
    // OpRegistry class
    py::class_<OpRegistry>(m, "OpRegistry")
        .def_static("instance", &OpRegistry::instance, py::return_value_policy::reference)
        .def("register_op", &OpRegistry::register_op)
        .def("get_op", &OpRegistry::get_op)
        .def("has_op", &OpRegistry::has_op)
        .def("get_forward_fn", &OpRegistry::get_forward_fn)
        .def("get_backward_fn", &OpRegistry::get_backward_fn);
    
    // Convenience function to register standard operations
    m.def("register_standard_operations", &register_standard_operations);
    
    // Convenience function to get operation from registry
    m.def("get_op", [](const std::string& name) {
        return OpRegistry::instance().get_op(name);
    });
    
    m.def("has_op", [](const std::string& name) {
        return OpRegistry::instance().has_op(name);
    });
    
    // Forward/backward function declarations for standard operations
    m.def("add_forward", &add_forward);
    m.def("add_backward", &add_backward);
    
    m.def("mul_forward", &mul_forward);
    m.def("mul_backward", &mul_backward);
    
    m.def("matmul_forward", &matmul_forward);
    m.def("matmul_backward", &matmul_backward);
    
    m.def("relu_forward", &relu_forward);
    m.def("relu_backward", &relu_backward);
    
    m.def("sigmoid_forward", &sigmoid_forward);
    m.def("sigmoid_backward", &sigmoid_backward);
    
    m.def("tanh_forward", &tanh_forward);
    m.def("tanh_backward", &tanh_backward);
    
    m.def("pow_forward", &pow_forward);
    m.def("pow_backward", &pow_backward);
    
    m.def("exp_forward", &exp_forward);
    m.def("exp_backward", &exp_backward);
    
    m.def("log_forward", &log_forward);
    m.def("log_backward", &log_backward);
    
    m.def("softmax_forward", &softmax_forward);
    m.def("softmax_backward", &softmax_backward);
}

} // namespace adaicpp
