#include "adaicpp/device.h"
#include <cstring>
#include <stdexcept>

#ifdef ADAICPP_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace adaicpp {

// CPUDevice Implementation
CPUDevice::CPUDevice(int device_id) : device_id_(device_id) {}

void* CPUDevice::allocate(size_t size) {
    void* ptr = std::malloc(size);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void CPUDevice::deallocate(void* ptr) {
    std::free(ptr);
}

void CPUDevice::memcpy(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
}

void CPUDevice::memcpy_host_to_device(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
}

void CPUDevice::memcpy_device_to_host(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
}

#ifdef ADAICPP_ENABLE_CUDA
// CUDADevice Implementation
CUDADevice::CUDADevice(int device_id) : device_id_(device_id) {
    cudaSetDevice(device_id);
}

CUDADevice::~CUDADevice() {
    // CUDA cleanup handled by CUDA runtime
}

void* CUDADevice::allocate(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA allocation failed: " + std::string(cudaGetErrorString(err)));
    }
    return ptr;
}

void CUDADevice::deallocate(void* ptr) {
    cudaFree(ptr);
}

void CUDADevice::memcpy(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
    }
}

void CUDADevice::memcpy_host_to_device(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA host to device memcpy failed: " + std::string(cudaGetErrorString(err)));
    }
}

void CUDADevice::memcpy_device_to_host(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA device to host memcpy failed: " + std::string(cudaGetErrorString(err)));
    }
}

std::string CUDADevice::name() const {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id_);
    return std::string(prop.name);
}
#endif

// DeviceManager Implementation
DeviceManager& DeviceManager::instance() {
    static DeviceManager instance;
    return instance;
}

std::shared_ptr<Device> DeviceManager::get_cpu_device(int device_id) {
    if (device_id >= static_cast<int>(cpu_devices_.size())) {
        cpu_devices_.resize(device_id + 1);
    }
    if (!cpu_devices_[device_id]) {
        cpu_devices_[device_id] = std::make_shared<CPUDevice>(device_id);
    }
    return cpu_devices_[device_id];
}

#ifdef ADAICPP_ENABLE_CUDA
std::shared_ptr<Device> DeviceManager::get_cuda_device(int device_id) {
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    
    if (device_id >= num_devices) {
        throw std::runtime_error("Invalid CUDA device ID");
    }
    
    if (device_id >= static_cast<int>(cuda_devices_.size())) {
        cuda_devices_.resize(device_id + 1);
    }
    if (!cuda_devices_[device_id]) {
        cuda_devices_[device_id] = std::make_shared<CUDADevice>(device_id);
    }
    return cuda_devices_[device_id];
}

int DeviceManager::num_cuda_devices() const {
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    return num_devices;
}
#endif

} // namespace adaicpp
