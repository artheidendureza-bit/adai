#include "adaicpp/graph.h"
#include "adaicpp/node.h"
#include "adaicpp/tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace adaicpp {

void bind_graph(py::module& m) {
    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        
        // Node management
        .def("add_node", &Graph::add_node)
        .def("remove_node", &Graph::remove_node)
        .def("get_node", &Graph::get_node)
        .def("has_node", &Graph::has_node)
        
        // Convenience methods
        .def("add_placeholder", &Graph::add_placeholder,
            py::arg("name"), py::arg("device") = nullptr)
        .def("add_parameter", &Graph::add_parameter,
            py::arg("name"), py::arg("value"))
        
        // Graph modification
        .def("set_inputs", &Graph::set_inputs)
        .def("modify_node_static", &Graph::modify_node_static)
        
        // Compilation and execution
        .def("compile", &Graph::compile)
        .def("forward", &Graph::forward, py::arg("feeds"))
        .def("backward", &Graph::backward, py::arg("target_node_name"))
        
        // Graph optimization
        .def("optimize", &Graph::optimize, py::arg("outputs"))
        
        // Accessors
        .def("get_sorted_nodes", &Graph::sorted_nodes)
        .def("is_compiled", &Graph::is_compiled)
        
        // Get node values and gradients
        .def("get_value", &Graph::get_value)
        .def("get_grad", &Graph::get_grad)
        
        // Clear gradients
        .def("zero_grad", &Graph::zero_grad)
        
        // Visualization
        .def("print_layout", &Graph::print_layout);
}

} // namespace adaicpp
