#pragma once

#include <memory>
#include <vector>
#include <cstddef>
#include <string>

namespace adaicpp {

enum class DeviceType {
    CPU,
    CUDA
};

class Device {
public:
    virtual ~Device() = default;
    
    virtual DeviceType type() const = 0;
    virtual void* allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void memcpy(void* dst, const void* src, size_t size) = 0;
    virtual void memcpy_host_to_device(void* dst, const void* src, size_t size) = 0;
    virtual void memcpy_device_to_host(void* dst, const void* src, size_t size) = 0;
    
    virtual int device_id() const = 0;
    virtual std::string name() const = 0;
};

class CPUDevice : public Device {
public:
    CPUDevice(int device_id = 0);
    
    DeviceType type() const override { return DeviceType::CPU; }
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void memcpy(void* dst, const void* src, size_t size) override;
    void memcpy_host_to_device(void* dst, const void* src, size_t size) override;
    void memcpy_device_to_host(void* dst, const void* src, size_t size) override;
    
    int device_id() const override { return device_id_; }
    std::string name() const override { return "CPU"; }
    
private:
    int device_id_;
};

#ifdef ADAICPP_ENABLE_CUDA
class CUDADevice : public Device {
public:
    CUDADevice(int device_id = 0);
    ~CUDADevice();
    
    DeviceType type() const override { return DeviceType::CUDA; }
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void memcpy(void* dst, const void* src, size_t size) override;
    void memcpy_host_to_device(void* dst, const void* src, size_t size) override;
    void memcpy_device_to_host(void* dst, const void* src, size_t size) override;
    
    int device_id() const override { return device_id_; }
    std::string name() const override;
    
private:
    int device_id_;
};
#endif

// Device manager for device pooling
class DeviceManager {
public:
    static DeviceManager& instance();
    
    std::shared_ptr<Device> get_cpu_device(int device_id = 0);
    
#ifdef ADAICPP_ENABLE_CUDA
    std::shared_ptr<Device> get_cuda_device(int device_id = 0);
    int num_cuda_devices() const;
#endif
    
private:
    DeviceManager() = default;
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    
    std::vector<std::shared_ptr<Device>> cpu_devices_;
#ifdef ADAICPP_ENABLE_CUDA
    std::vector<std::shared_ptr<Device>> cuda_devices_;
#endif
};

} // namespace adaicpp
