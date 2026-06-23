#include "adaicpp/node.h"
#include "adaicpp/tensor.h"
#include "adaicpp/device.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace adaicpp {

void bind_node(py::module& m) {
    // NodeType enum
    py::enum_<NodeType>(m, "NodeType")
        .value("Base", NodeType::Base)
        .value("Placeholder", NodeType::Placeholder)
        .value("Parameter", NodeType::Parameter)
        .value("Operation", NodeType::Operation);
    
    // Node base class
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def_property_readonly("name", &Node::name)
        .def_property_readonly("type", &Node::type)
        .def_property("is_static", &Node::is_static, &Node::set_is_static)
        .def_property_readonly("device", &Node::device)
        
        .def("get_value", static_cast<std::shared_ptr<Tensor> (Node::*)()>(&Node::value))
        .def("set_value", &Node::set_value)
        .def("get_grad", static_cast<std::shared_ptr<Tensor> (Node::*)()>(&Node::grad))
        .def("set_grad", &Node::set_grad)
        
        .def("get_inputs", static_cast<const std::vector<std::shared_ptr<Node>>& (Node::*)() const>(&Node::inputs))
        .def("get_consumers", static_cast<const std::vector<std::weak_ptr<Node>>& (Node::*)() const>(&Node::consumers))
        
        .def_property_readonly("is_dirty", &Node::is_dirty)
        .def_property_readonly("eval_count", &Node::eval_count)
        
        .def("evaluate", &Node::evaluate)
        .def("backward", &Node::backward)
        .def("mark_dirty", &Node::mark_dirty);
    
    // PlaceholderNode
    py::class_<PlaceholderNode, Node, std::shared_ptr<PlaceholderNode>>(m, "PlaceholderNode")
        .def(py::init<const std::string&, std::shared_ptr<Device>>(),
            py::arg("name"), py::arg("device") = nullptr)
        .def("set_value", &PlaceholderNode::set_value);
    
    // ParameterNode
    py::class_<ParameterNode, Node, std::shared_ptr<ParameterNode>>(m, "ParameterNode")
        .def(py::init<const std::string&, std::shared_ptr<Tensor>>(),
            py::arg("name"), py::arg("value"))
        .def("set_value", &ParameterNode::set_value);
    
    // OpNode
    py::class_<OpNode, Node, std::shared_ptr<OpNode>>(m, "OpNode")
        .def(py::init<const std::string&, 
                      const std::vector<std::shared_ptr<Node>>&,
                      OpForwardFn,
                      OpBackwardFn,
                      bool,
                      std::shared_ptr<Device>>(),
            py::arg("name"), py::arg("inputs"), py::arg("forward_fn"),
            py::arg("backward_fn") = nullptr, py::arg("is_static") = true,
            py::arg("device") = nullptr)
        .def_property_readonly("op_name", &OpNode::op_name)
        .def("set_op_name", &OpNode::set_op_name)
        .def("register_consumers", &OpNode::register_consumers);
    
    // Helper to create Python-callable forward functions
    m.def("make_forward_fn", [](py::function func) {
        return OpForwardFn([func](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            py::list py_inputs;
            for (const auto& input : inputs) {
                py_inputs.append(input);
            }
            py::object result = func(*py_inputs);
            return result.cast<std::shared_ptr<Tensor>>();
        });
    });
    
    // Helper to create Python-callable backward functions
    m.def("make_backward_fn", [](py::function func) {
        return OpBackwardFn([func](const std::shared_ptr<Tensor>& out_grad,
                                    const std::vector<std::shared_ptr<Tensor>>& inputs) {
            py::list py_inputs;
            for (const auto& input : inputs) {
                py_inputs.append(input);
            }
            py::object result = func(out_grad, *py_inputs);
            return result.cast<std::vector<std::shared_ptr<Tensor>>>();
        });
    });
}

} // namespace adaicpp
