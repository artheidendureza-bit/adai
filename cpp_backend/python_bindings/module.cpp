#include "adaicpp/adaicpp.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations for binding functions
namespace adaicpp {
    void bind_tensor(py::module& m);
    void bind_device(py::module& m);
    void bind_node(py::module& m);
    void bind_graph(py::module& m);
    void bind_op_registry(py::module& m);
}

PYBIND11_MODULE(adaicpp, m) {
    m.doc() = "adAI C++ Backend - High-performance tensor operations and computational graph";
    
    // Initialize the C++ library
    adaicpp::initialize();
    
    // Bind all components
    adaicpp::bind_device(m);
    adaicpp::bind_tensor(m);
    adaicpp::bind_node(m);
    adaicpp::bind_graph(m);
    adaicpp::bind_op_registry(m);
    
    // Module-level convenience functions
    m.def("initialize", &adaicpp::initialize, "Initialize the adAI C++ backend");
}
