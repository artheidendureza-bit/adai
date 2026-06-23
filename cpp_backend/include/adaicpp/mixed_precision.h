#pragma once

#include "adaicpp/tensor.h"
#include "adaicpp/device.h"
#include "adaicpp/model.h"
#include <memory>
#include <vector>

#ifdef ADAICPP_ENABLE_CUDA
#include <cuda_fp16.h>
#endif

namespace adaicpp {

enum class Precision {
    FP32,
    FP16,
    MIXED
};

class MixedPrecisionTrainer {
public:
    MixedPrecisionTrainer(std::shared_ptr<Model> model, 
                        float loss_scale = 1.0f);
    
    // Convert model to mixed precision
    void convert_to_mixed_precision();
    
    // Training step with automatic loss scaling
    void training_step(const std::vector<std::shared_ptr<Tensor>>& x_batch,
                     const std::shared_ptr<Tensor>& y_batch);
    
    // Get current loss scale
    float get_loss_scale() const { return loss_scale_; }
    
    // Update loss scale (for dynamic loss scaling)
    void update_loss_scale(bool overflow_occurred);
    
private:
    std::shared_ptr<Model> model_;
    float loss_scale_;
    
    // Store FP32 master weights
    std::vector<std::shared_ptr<Tensor>> master_weights_;
    
    // FP16 working weights
    std::vector<std::shared_ptr<Tensor>> fp16_weights_;
    
    // Helper methods
    void initialize_master_weights();
    void convert_weights_to_fp16();
    void convert_weights_to_fp32();
    
#ifdef ADAICPP_ENABLE_CUDA
    void convert_tensor_fp32_to_fp16(const Tensor& fp32, Tensor& fp16);
    void convert_tensor_fp16_to_fp32(const Tensor& fp16, Tensor& fp32);
#endif
};

// Utility functions for precision conversion
Tensor convert_precision(const Tensor& tensor, Precision target_precision);

// Check if device supports mixed precision
bool supports_mixed_precision(std::shared_ptr<Device> device);

} // namespace adaicpp
