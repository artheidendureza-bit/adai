#include "adaicpp/mixed_precision.h"
#include "adaicpp/tensor.h"
#include "adaicpp/device.h"
#include "adaicpp/model.h"
#ifdef ADAICPP_ENABLE_CUDA
#include "adaicpp/cuda_kernels.h"
#endif
#include <stdexcept>

namespace adaicpp {

MixedPrecisionTrainer::MixedPrecisionTrainer(std::shared_ptr<Model> model, 
                                            float loss_scale)
    : model_(model), loss_scale_(loss_scale) {
    
    if (!supports_mixed_precision(model_->graph()->get_node("input")->device())) {
        throw std::runtime_error("Device does not support mixed precision training");
    }
    
    initialize_master_weights();
}

void MixedPrecisionTrainer::initialize_master_weights() {
    // Store FP32 master weights for all parameters
    for (const auto& [name, node] : model_->graph()->nodes()) {
        if (node->type() == NodeType::Parameter) {
            auto param_node = std::dynamic_pointer_cast<ParameterNode>(node);
            if (param_node && param_node->value()) {
                master_weights_.push_back(param_node->value());
            }
        }
    }
}

void MixedPrecisionTrainer::convert_to_mixed_precision() {
    convert_weights_to_fp16();
}

void MixedPrecisionTrainer::convert_weights_to_fp16() {
#ifdef ADAICPP_ENABLE_CUDA
    fp16_weights_.clear();
    
    for (auto& master_weight : master_weights_) {
        // Create FP16 tensor
        auto fp16_tensor = std::make_shared<Tensor>(master_weight->shape(), master_weight->device());
        convert_tensor_fp32_to_fp16(*master_weight, *fp16_tensor);
        fp16_weights_.push_back(fp16_tensor);
    }
#else
    throw std::runtime_error("Mixed precision requires CUDA support");
#endif
}

void MixedPrecisionTrainer::convert_weights_to_fp32() {
#ifdef ADAICPP_ENABLE_CUDA
    for (size_t i = 0; i < master_weights_.size(); i++) {
        convert_tensor_fp16_to_fp32(*fp16_weights_[i], *master_weights_[i]);
    }
#else
    throw std::runtime_error("Mixed precision requires CUDA support");
#endif
}

void MixedPrecisionTrainer::training_step(const std::vector<std::shared_ptr<Tensor>>& x_batch,
                                         const std::shared_ptr<Tensor>& y_batch) {
#ifdef ADAICPP_ENABLE_CUDA
    // Convert to FP16 for forward pass
    convert_weights_to_fp16();
    
    // Forward pass with FP16
    // (This would require updating the model to use fp16_weights_)
    
    // Scale loss
    // (Loss scaling to prevent underflow)
    
    // Backward pass with scaled gradients
    
    // Convert gradients back to FP32 and unscale
    
    // Update master weights
    
    // Check for overflow and adjust loss scale
#else
    throw std::runtime_error("Mixed precision requires CUDA support");
#endif
}

void MixedPrecisionTrainer::update_loss_scale(bool overflow_occurred) {
    if (overflow_occurred) {
        loss_scale_ *= 0.5f;  // Reduce loss scale
        if (loss_scale_ < 1e-4f) {
            loss_scale_ = 1e-4f;  // Minimum loss scale
        }
    } else {
        loss_scale_ *= 2.0f;  // Increase loss scale
        if (loss_scale_ > 65536.0f) {
            loss_scale_ = 65536.0f;  // Maximum loss scale
        }
    }
}

#ifdef ADAICPP_ENABLE_CUDA
void MixedPrecisionTrainer::convert_tensor_fp32_to_fp16(const Tensor& fp32, Tensor& fp16) {
    if (fp32.device()->type() != DeviceType::CUDA) {
        throw std::runtime_error("FP16 conversion requires CUDA device");
    }
    
    // Allocate FP16 memory
    __half* fp16_data = static_cast<__half*>(fp32.device()->allocate(fp32.size() * sizeof(__half)));
    
    // Convert
    cuda::launch_fp32_to_fp16(fp32.data(), fp16_data, fp32.size());
    
    // Set FP16 tensor data (this would need proper Tensor support for FP16)
    // For now, this is a placeholder
}

void MixedPrecisionTrainer::convert_tensor_fp16_to_fp32(const Tensor& fp16, Tensor& fp32) {
    if (fp16.device()->type() != DeviceType::CUDA) {
        throw std::runtime_error("FP16 conversion requires CUDA device");
    }
    
    // Convert FP16 to FP32
    // This would need proper Tensor support for FP16
    // For now, this is a placeholder
}
#endif

Tensor convert_precision(const Tensor& tensor, Precision target_precision) {
    if (target_precision == Precision::FP32) {
        return tensor;  // Already FP32
    } else if (target_precision == Precision::FP16) {
#ifdef ADAICPP_ENABLE_CUDA
        if (tensor.device()->type() == DeviceType::CUDA) {
            // Convert to FP16
            Tensor result(tensor.shape(), tensor.device());
            // Conversion would happen here
            return result;
        }
#endif
        throw std::runtime_error("FP16 requires CUDA support");
    }
    return tensor;
}

bool supports_mixed_precision(std::shared_ptr<Device> device) {
    (void)device;
#ifdef ADAICPP_ENABLE_CUDA
    return device->type() == DeviceType::CUDA;
#else
    return false;
#endif
}

} // namespace adaicpp
