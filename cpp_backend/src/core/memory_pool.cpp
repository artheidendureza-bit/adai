#include "adaicpp/memory_pool.h"
#include <algorithm>

namespace adaicpp {

// MemoryBlock Implementation
MemoryBlock::MemoryBlock(void* ptr, size_t size, std::shared_ptr<Device> device)
    : ptr_(ptr), size_(size), device_(device) {}

MemoryBlock::~MemoryBlock() {
    if (ptr_) {
        device_->deallocate(ptr_);
    }
}

// MemoryPool Implementation
MemoryPool& MemoryPool::instance() {
    static MemoryPool instance;
    return instance;
}

void* MemoryPool::allocate(size_t size, std::shared_ptr<Device> device) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    DeviceCache& cache = device_caches_[device.get()];
    
    // Try to find a free block
    MemoryBlock* block = find_free_block(cache, size);
    
    if (block) {
        // Cache hit
        cache_hits_++;
        cache.total_cached -= block->size();
        
        void* ptr = block->ptr();
        cache.allocated_sizes[ptr] = block->size();
        cache.total_allocated += block->size();
        
        // Remove from free list
        auto it = std::find_if(cache.free_blocks.begin(), cache.free_blocks.end(),
            [block](const auto& b) { return b.get() == block; });
        if (it != cache.free_blocks.end()) {
            cache.free_blocks.erase(it);
        }
        
        return ptr;
    }
    
    // Cache miss - allocate new block
    cache_misses_++;
    void* ptr = device->allocate(size);
    cache.allocated_sizes[ptr] = size;
    cache.total_allocated += size;
    
    return ptr;
}

void MemoryPool::deallocate(void* ptr, std::shared_ptr<Device> device) {
    if (!ptr) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    DeviceCache& cache = device_caches_[device.get()];
    
    auto it = cache.allocated_sizes.find(ptr);
    if (it == cache.allocated_sizes.end()) {
        // Not allocated through pool, deallocate directly
        device->deallocate(ptr);
        return;
    }
    
    size_t size = it->second;
    cache.allocated_sizes.erase(it);
    cache.total_allocated -= size;
    
    // Return to pool
    cache.free_blocks.push_back(std::make_unique<MemoryBlock>(ptr, size, device));
    cache.total_cached += size;
}

void MemoryPool::clear_device_cache(std::shared_ptr<Device> device) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = device_caches_.find(device.get());
    if (it != device_caches_.end()) {
        it->second.free_blocks.clear();
        it->second.total_cached = 0;
    }
}

void MemoryPool::clear_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [device_ptr, cache] : device_caches_) {
        cache.free_blocks.clear();
        cache.total_cached = 0;
    }
}

size_t MemoryPool::total_allocated_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t total = 0;
    for (const auto& [device_ptr, cache] : device_caches_) {
        total += cache.total_allocated;
    }
    return total;
}

size_t MemoryPool::total_cached_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t total = 0;
    for (const auto& [device_ptr, cache] : device_caches_) {
        total += cache.total_cached;
    }
    return total;
}

size_t MemoryPool::cache_hits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_hits_;
}

size_t MemoryPool::cache_misses() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_misses_;
}

MemoryBlock* MemoryPool::find_free_block(DeviceCache& cache, size_t size) {
    // Find the smallest block that fits
    MemoryBlock* best_block = nullptr;
    size_t best_size = SIZE_MAX;
    
    for (auto& block : cache.free_blocks) {
        if (block->size() >= size && block->size() < best_size) {
            best_block = block.get();
            best_size = block->size();
            
            // If we find an exact match, use it immediately
            if (block->size() == size) {
                break;
            }
        }
    }
    
    if (best_block) {
        // Consider splitting if the block is much larger than needed
        if (best_size > size * 2) {
            split_block(cache, best_block, size);
        }
    }
    
    return best_block;
}

void MemoryPool::split_block(DeviceCache& cache, MemoryBlock* block, size_t requested_size) {
    // For simplicity, we don't actually split blocks in this implementation
    // In a more sophisticated implementation, we would split the block
    // and return the unused portion to the free list
}

// PooledMemory Implementation
PooledMemory::PooledMemory(size_t size, std::shared_ptr<Device> device)
    : size_(size), device_(device) {
    ptr_ = MemoryPool::instance().allocate(size, device);
}

PooledMemory::~PooledMemory() {
    if (ptr_) {
        MemoryPool::instance().deallocate(ptr_, device_);
    }
}

PooledMemory::PooledMemory(PooledMemory&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_), device_(std::move(other.device_)) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

PooledMemory& PooledMemory::operator=(PooledMemory&& other) noexcept {
    if (this != &other) {
        if (ptr_) {
            MemoryPool::instance().deallocate(ptr_, device_);
        }
        
        ptr_ = other.ptr_;
        size_ = other.size_;
        device_ = std::move(other.device_);
        
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

} // namespace adaicpp
