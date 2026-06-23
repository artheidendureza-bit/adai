#pragma once

#include "adaicpp/device.h"
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <cstddef>

namespace adaicpp {

class MemoryBlock {
public:
    MemoryBlock(void* ptr, size_t size, std::shared_ptr<Device> device);
    ~MemoryBlock();
    
    void* ptr() const { return ptr_; }
    size_t size() const { return size_; }
    std::shared_ptr<Device> device() const { return device_; }
    
private:
    void* ptr_;
    size_t size_;
    std::shared_ptr<Device> device_;
};

class MemoryPool {
public:
    static MemoryPool& instance();
    
    // Allocate memory from pool or device
    void* allocate(size_t size, std::shared_ptr<Device> device);
    
    // Return memory to pool
    void deallocate(void* ptr, std::shared_ptr<Device> device);
    
    // Clear all cached memory for a specific device
    void clear_device_cache(std::shared_ptr<Device> device);
    
    // Clear all cached memory
    void clear_all();
    
    // Get statistics
    size_t total_allocated_bytes() const;
    size_t total_cached_bytes() const;
    size_t cache_hits() const;
    size_t cache_misses() const;
    
private:
    MemoryPool() = default;
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    struct DeviceCache {
        std::vector<std::unique_ptr<MemoryBlock>> free_blocks;
        std::unordered_map<void*, size_t> allocated_sizes;
        size_t total_allocated = 0;
        size_t total_cached = 0;
    };
    
    std::unordered_map<Device*, DeviceCache> device_caches_;
    mutable std::mutex mutex_;
    
    size_t cache_hits_ = 0;
    size_t cache_misses_ = 0;
    
    // Find a free block of at least the requested size
    MemoryBlock* find_free_block(DeviceCache& cache, size_t size);
    
    // Split a block if it's significantly larger than requested
    void split_block(DeviceCache& cache, MemoryBlock* block, size_t requested_size);
};

// RAII wrapper for pooled memory
class PooledMemory {
public:
    PooledMemory(size_t size, std::shared_ptr<Device> device);
    ~PooledMemory();
    
    void* ptr() const { return ptr_; }
    size_t size() const { return size_; }
    std::shared_ptr<Device> device() const { return device_; }
    
    // Disable copy
    PooledMemory(const PooledMemory&) = delete;
    PooledMemory& operator=(const PooledMemory&) = delete;
    
    // Enable move
    PooledMemory(PooledMemory&& other) noexcept;
    PooledMemory& operator=(PooledMemory&& other) noexcept;
    
private:
    void* ptr_;
    size_t size_;
    std::shared_ptr<Device> device_;
};

} // namespace adaicpp
