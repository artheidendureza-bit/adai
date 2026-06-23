#include "adaicpp/tensor.h"
#include "adaicpp/device.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace adaicpp {

// Helper function to convert numpy array to Tensor
Tensor numpy_to_tensor(py::array_t<float> array, std::shared_ptr<Device> device = nullptr) {
    py::buffer_info buf = array.request();
    
    if (buf.ndim == 0) {
        throw std::runtime_error("Cannot convert 0-dimensional array");
    }
    
    std::vector<size_t> shape(buf.ndim);
    for (int i = 0; i < buf.ndim; ++i) {
        shape[i] = buf.shape[i];
    }
    
    Tensor tensor(shape, device);
    
    if (device && device->type() == DeviceType::CPU) {
        float* data = static_cast<float*>(buf.ptr);
        std::copy(data, data + tensor.size(), tensor.data());
    } else if (!device) {
        auto cpu_device = DeviceManager::instance().get_cpu_device();
        float* data = static_cast<float*>(buf.ptr);
        std::copy(data, data + tensor.size(), tensor.data());
    } else {
        throw std::runtime_error("CUDA numpy conversion not yet implemented");
    }
    
    return tensor;
}

// Helper function to convert Tensor to numpy array
py::array_t<float> tensor_to_numpy(const Tensor& tensor) {
    if (tensor.device() && tensor.device()->type() != DeviceType::CPU) {
        throw std::runtime_error("Can only convert CPU tensors to numpy");
    }
    
    std::vector<py::ssize_t> shape(tensor.ndim());
    for (size_t i = 0; i < tensor.ndim(); ++i) {
        shape[i] = static_cast<py::ssize_t>(tensor.shape()[i]);
    }
    
    return py::array_t<float>(shape, tensor.data());
}

void bind_tensor(py::module& m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<size_t>&, std::shared_ptr<Device>>(),
            py::arg("shape"), py::arg("device") = nullptr)
        .def(py::init<const std::vector<size_t>&, float, std::shared_ptr<Device>>(),
            py::arg("shape"), py::arg("value"), py::arg("device") = nullptr)
        .def(py::init<const std::vector<size_t>&, const std::vector<float>&, std::shared_ptr<Device>>(),
            py::arg("shape"), py::arg("data"), py::arg("device") = nullptr)
        
        // Accessors
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("size", &Tensor::size)
        .def_property_readonly("strides", &Tensor::strides)
        .def_property("device", &Tensor::device, &Tensor::to_device)
        
        // Data access
        .def("numpy", [](Tensor& self) {
            return tensor_to_numpy(self);
        })
        
        // Utility functions
        .def("fill", &Tensor::fill)
        .def("zero", &Tensor::zero)
        .def("copy_from", &Tensor::copy_from)
        .def("is_contiguous", &Tensor::is_contiguous)
        
        // Reshape and view operations
        .def("reshape", &Tensor::reshape)
        .def("view", &Tensor::view)
        .def("transpose", &Tensor::transpose)
        
        // Gradient tracking
        .def_property("requires_grad", &Tensor::requires_grad, &Tensor::set_requires_grad)
        .def_property_readonly("grad", static_cast<Tensor* (Tensor::*)()>(&Tensor::grad))
        .def("set_grad", &Tensor::set_grad)
        .def("zero_grad", &Tensor::zero_grad)
        
        // Static methods
        .def_static("zeros", &Tensor::zeros, py::arg("shape"), py::arg("device") = nullptr)
        .def_static("ones", &Tensor::ones, py::arg("shape"), py::arg("device") = nullptr)
        .def_static("randn", &Tensor::randn, py::arg("shape"), py::arg("device") = nullptr)
        .def_static("arange", &Tensor::arange, 
            py::arg("start"), py::arg("stop"), py::arg("step") = 1.0f, py::arg("device") = nullptr);
    
    // Tensor operations
    m.def("add", &add);
    m.def("sub", &sub);
    m.def("mul", &mul);
    m.def("div", &div);
    m.def("pow", &pow);
    m.def("exp", &exp);
    m.def("log", &log);
    m.def("scalar_mul", &scalar_mul);
    
    m.def("relu", &relu);
    m.def("sigmoid", &sigmoid);
    m.def("tanh", &tanh);
    m.def("softmax", &softmax);
    
    m.def("matmul", &matmul);
    
    m.def("conv2d", &conv2d,
        py::arg("input"), py::arg("kernel"),
        py::arg("stride_h") = 1, py::arg("stride_w") = 1,
        py::arg("padding_h") = 0, py::arg("padding_w") = 0);
    
    m.def("sum", &sum, py::arg("x"), py::arg("axes") = std::vector<size_t>{}, py::arg("keepdims") = false);
    m.def("mean", &mean, py::arg("x"), py::arg("axes") = std::vector<size_t>{}, py::arg("keepdims") = false);
    m.def("max", &max, py::arg("x"), py::arg("axes") = std::vector<size_t>{}, py::arg("keepdims") = false);
    m.def("min", &min, py::arg("x"), py::arg("axes") = std::vector<size_t>{}, py::arg("keepdims") = false);
    
    m.def("transpose", &transpose);
    m.def("pad", &pad, py::arg("x"), py::arg("padding"), py::arg("value") = 0.0f);
    
    // Numpy conversion helpers
    m.def("from_numpy", &numpy_to_tensor, py::arg("array"), py::arg("device") = nullptr);
}

} // namespace adaicpp
