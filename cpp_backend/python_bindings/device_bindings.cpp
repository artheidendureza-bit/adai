#include "adaicpp/device.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace adaicpp {

void bind_device(py::module& m) {
    // DeviceType enum
    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA);
    
    // Device base class
    py::class_<Device, std::shared_ptr<Device>>(m, "Device")
        .def("type", &Device::type)
        .def("device_id", &Device::device_id)
        .def("name", &Device::name);
    
    // CPUDevice
    py::class_<CPUDevice, Device, std::shared_ptr<CPUDevice>>(m, "CPUDevice")
        .def(py::init<int>(), py::arg("device_id") = 0);
    
#ifdef ADAICPP_ENABLE_CUDA
    // CUDADevice
    py::class_<CUDADevice, Device, std::shared_ptr<CUDADevice>>(m, "CUDADevice")
        .def(py::init<int>(), py::arg("device_id") = 0);
#endif
    
    // DeviceManager
    py::class_<DeviceManager>(m, "DeviceManager")
        .def_static("instance", &DeviceManager::instance, py::return_value_policy::reference)
        .def("get_cpu_device", &DeviceManager::get_cpu_device, py::arg("device_id") = 0)
#ifdef ADAICPP_ENABLE_CUDA
        .def("get_cuda_device", &DeviceManager::get_cuda_device, py::arg("device_id") = 0)
        .def("num_cuda_devices", &DeviceManager::num_cuda_devices)
#endif
    ;
    
    // Convenience function to get CPU device
    m.def("cpu_device", []() {
        return DeviceManager::instance().get_cpu_device();
    });
    
#ifdef ADAICPP_ENABLE_CUDA
    m.def("cuda_device", [](int device_id = 0) {
        return DeviceManager::instance().get_cuda_device(device_id);
    }, py::arg("device_id") = 0);
#endif
}

} // namespace adaicpp
