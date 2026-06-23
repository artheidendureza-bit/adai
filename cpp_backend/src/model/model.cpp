#include "adaicpp/model.h"
#include "adaicpp/tensor.h"
#include "adaicpp/device.h"
#include "adaicpp/op_registry.h"
#include <stdexcept>
#include <random>
#include <algorithm>

namespace adaicpp {

// === Dense Layer ===
DenseLayer::DenseLayer(int input_size, int output_size, 
                       const std::string& activation,
                       const std::string& name)
    : input_size_(input_size), output_size_(output_size),
      activation_(activation), name_(name) {
    
    auto device = DeviceManager::instance().get_cpu_device();
    
    // Initialize weights with Xavier initialization
    float scale = std::sqrt(2.0f / (input_size + output_size));
    weights_ = std::make_shared<Tensor>(scalar_mul(Tensor::randn({static_cast<size_t>(input_size), static_cast<size_t>(output_size)}, device), scale));
    bias_ = std::make_shared<Tensor>(Tensor::zeros({static_cast<size_t>(output_size)}, device));
}

std::vector<std::shared_ptr<Tensor>> DenseLayer::parameters() const {
    return {weights_, bias_};
}

void DenseLayer::set_parameters(const std::vector<std::shared_ptr<Tensor>>& params) {
    if (params.size() != 2) {
        throw std::runtime_error("DenseLayer requires 2 parameters (weights, bias)");
    }
    weights_ = params[0];
    bias_ = params[1];
}

std::shared_ptr<Node> DenseLayer::build(std::shared_ptr<Graph> graph, 
                                       std::shared_ptr<Node> input) {
    auto device = DeviceManager::instance().get_cpu_device();
    
    // Add parameters to graph
    auto w_param = graph->add_parameter(name_ + "_weights", weights_);
    auto b_param = graph->add_parameter(name_ + "_bias", bias_);
    
    // Matmul operation
    auto matmul_node = graph->add_op(name_ + "_matmul", "matmul", {input, w_param});
    
    // Add bias
    auto add_node = graph->add_op(name_ + "_add", "add", {matmul_node, b_param});
    
    // Activation
    if (activation_ == "relu") {
        return graph->add_op(name_ + "_relu", "relu", {add_node});
    } else if (activation_ == "sigmoid") {
        return graph->add_op(name_ + "_sigmoid", "sigmoid", {add_node});
    } else if (activation_ == "tanh") {
        return graph->add_op(name_ + "_tanh", "tanh", {add_node});
    } else if (activation_ == "linear" || activation_ == "none") {
        return add_node;
    } else {
        throw std::runtime_error("Unknown activation: " + activation_);
    }
}

// === Conv2D Layer ===
Conv2DLayer::Conv2DLayer(int in_channels, int out_channels, 
                        int kernel_size, int stride, int padding,
                        const std::string& activation,
                        const std::string& name)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      activation_(activation), name_(name) {
    
    auto device = DeviceManager::instance().get_cpu_device();
    
    // Initialize weights with Xavier initialization
    float scale = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    weights_ = std::make_shared<Tensor>(
        scalar_mul(Tensor::randn({static_cast<size_t>(out_channels), static_cast<size_t>(in_channels), 
                      static_cast<size_t>(kernel_size), static_cast<size_t>(kernel_size)}, device), scale)
    );
    bias_ = std::make_shared<Tensor>(Tensor::zeros({static_cast<size_t>(out_channels)}, device));
}

std::vector<std::shared_ptr<Tensor>> Conv2DLayer::parameters() const {
    return {weights_, bias_};
}

void Conv2DLayer::set_parameters(const std::vector<std::shared_ptr<Tensor>>& params) {
    if (params.size() != 2) {
        throw std::runtime_error("Conv2DLayer requires 2 parameters (weights, bias)");
    }
    weights_ = params[0];
    bias_ = params[1];
}

std::shared_ptr<Node> Conv2DLayer::build(std::shared_ptr<Graph> graph, 
                                        std::shared_ptr<Node> input) {
    auto device = DeviceManager::instance().get_cpu_device();
    
    // Add parameters to graph
    auto w_param = graph->add_parameter(name_ + "_weights", weights_);
    auto b_param = graph->add_parameter(name_ + "_bias", bias_);
    
    // Conv2D operation (custom lambda for now)
    auto conv_node = graph->add_op(name_ + "_conv",
        {input, w_param},
        [this](const std::vector<std::shared_ptr<Tensor>>& inputs) -> std::shared_ptr<Tensor> {
            return std::make_shared<Tensor>(conv2d(*inputs[0], *inputs[1], stride_, stride_, padding_, padding_));
        },
        nullptr, true, device);
    
    // Add bias (broadcast)
    auto add_node = graph->add_op(name_ + "_add", "add", {conv_node, b_param});
    
    // Activation
    if (activation_ == "relu") {
        return graph->add_op(name_ + "_relu", "relu", {add_node});
    } else if (activation_ == "sigmoid") {
        return graph->add_op(name_ + "_sigmoid", "sigmoid", {add_node});
    } else if (activation_ == "tanh") {
        return graph->add_op(name_ + "_tanh", "tanh", {add_node});
    } else if (activation_ == "linear" || activation_ == "none") {
        return add_node;
    } else {
        throw std::runtime_error("Unknown activation: " + activation_);
    }
}

// === MaxPool2D Layer ===
MaxPool2DLayer::MaxPool2DLayer(int pool_size, int stride, 
                              const std::string& name)
    : pool_size_(pool_size), stride_(stride < 0 ? pool_size : stride),
      name_(name) {}

std::shared_ptr<Node> MaxPool2DLayer::build(std::shared_ptr<Graph> graph, 
                                          std::shared_ptr<Node> input) {
    auto device = DeviceManager::instance().get_cpu_device();
    
    // MaxPool operation (custom lambda for now)
    return graph->add_op(name_,
        {input},
        [this](const std::vector<std::shared_ptr<Tensor>>& inputs) -> std::shared_ptr<Tensor> {
            // Simple max pool implementation
            auto& input_tensor = inputs[0];
            // TODO: Implement actual max pooling
            return input_tensor; // Placeholder
        },
        nullptr, true, device);
}

// === Flatten Layer ===
FlattenLayer::FlattenLayer(const std::string& name) : name_(name) {}

std::shared_ptr<Node> FlattenLayer::build(std::shared_ptr<Graph> graph, 
                                        std::shared_ptr<Node> input) {
    auto device = DeviceManager::instance().get_cpu_device();
    
    // Flatten operation (custom lambda)
    return graph->add_op(name_,
        {input},
        [](const std::vector<std::shared_ptr<Tensor>>& inputs) -> std::shared_ptr<Tensor> {
            auto& input_tensor = inputs[0];
            std::vector<size_t> new_shape = {input_tensor->size()};
            return std::make_shared<Tensor>(input_tensor->reshape(new_shape));
        },
        nullptr, true, device);
}

// === Model ===
Model::Model(const std::string& name)
    : name_(name), built_(false), compiled_(false) {
    graph_ = std::make_shared<Graph>();
}

void Model::add_layer(std::shared_ptr<Layer> layer) {
    layers_.push_back(layer);
}

void Model::build(const std::vector<size_t>& input_shape) {
    input_shape_ = input_shape;
    
    // Create input placeholder
    input_node_ = graph_->add_placeholder(name_ + "_input");
    
    // Build layers sequentially
    std::shared_ptr<Node> current = input_node_;
    for (auto& layer : layers_) {
        current = layer->build(graph_, current);
    }
    
    output_node_ = current;
    built_ = true;
}

void Model::compile(const std::string& optimizer, 
                   float learning_rate,
                   const std::string& loss) {
    if (!built_) {
        throw std::runtime_error("Model must be built before compiling");
    }
    
    optimizer_ = optimizer;
    learning_rate_ = learning_rate;
    loss_ = loss;
    
    // Compile the graph
    graph_->compile();
    compiled_ = true;
}

void Model::fit(
    const std::vector<std::shared_ptr<Tensor>>& x_train,
    const std::vector<std::shared_ptr<Tensor>>& y_train,
    int epochs,
    int batch_size,
    const std::vector<std::shared_ptr<Tensor>>& x_val,
    const std::vector<std::shared_ptr<Tensor>>& y_val,
    std::vector<std::shared_ptr<TrainingCallback>> callbacks
) {
    if (!compiled_) {
        throw std::runtime_error("Model must be compiled before training");
    }
    
    int num_samples = x_train.size();
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        // Call epoch begin callbacks
        for (auto& callback : callbacks) {
            callback->on_epoch_begin(epoch);
        }
        
        // Shuffle training data
        std::vector<int> indices(num_samples);
        for (int i = 0; i < num_samples; i++) indices[i] = i;
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Process batches
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, num_samples);
            
            // Call batch begin callbacks
            for (auto& callback : callbacks) {
                callback->on_batch_begin(batch, num_batches);
            }
            
            // Forward pass
            std::unordered_map<std::string, std::shared_ptr<Tensor>> feeds;
            feeds[name_ + "_input"] = x_train[indices[start_idx]];
            
            graph_->forward(feeds);
            
            // Compute loss
            auto output = graph_->get_value(output_node_->name());
            auto target = y_train[indices[start_idx]];
            
            // Simple MSE loss
            auto diff = sub(*output, *target);
            auto squared = mul(diff, diff);
            auto loss_tensor = sum(squared);
            
            float batch_loss = loss_tensor.data()[0] / output->size();
            epoch_loss += batch_loss;
            
            // Backward pass
            graph_->backward(output_node_->name());
            
            // Update parameters (simple SGD for now)
            update_parameters();
            
            // Call batch end callbacks
            for (auto& callback : callbacks) {
                callback->on_batch_end(batch, batch_loss);
            }
        }
        
        epoch_loss /= num_batches;
        loss_history_.push_back(epoch_loss);
        
        // Validation
        float val_loss = 0.0f;
        if (!x_val.empty() && !y_val.empty()) {
            for (size_t i = 0; i < x_val.size(); i++) {
                std::unordered_map<std::string, std::shared_ptr<Tensor>> feeds;
                feeds[name_ + "_input"] = x_val[i];
                
                graph_->forward(feeds);
                auto output = graph_->get_value(output_node_->name());
                auto target = y_val[i];
                
                auto diff = sub(*output, *target);
                auto squared = mul(diff, diff);
                auto loss_tensor = sum(squared);
                val_loss += loss_tensor.data()[0] / output->size();
            }
            val_loss /= x_val.size();
            val_loss_history_.push_back(val_loss);
        }
        
        // Call epoch end callbacks
        for (auto& callback : callbacks) {
            callback->on_epoch_end(epoch, epoch_loss, val_loss);
        }
    }
}

std::shared_ptr<Tensor> Model::predict(std::shared_ptr<Tensor> x) {
    if (!compiled_) {
        throw std::runtime_error("Model must be compiled before prediction");
    }
    
    std::unordered_map<std::string, std::shared_ptr<Tensor>> feeds;
    feeds[name_ + "_input"] = x;
    
    graph_->forward(feeds);
    return graph_->get_value(output_node_->name());
}

float Model::evaluate(const std::vector<std::shared_ptr<Tensor>>& x_test,
                     const std::vector<std::shared_ptr<Tensor>>& y_test) {
    if (!compiled_) {
        throw std::runtime_error("Model must be compiled before evaluation");
    }
    
    float total_loss = 0.0f;
    for (size_t i = 0; i < x_test.size(); i++) {
        std::unordered_map<std::string, std::shared_ptr<Tensor>> feeds;
        feeds[name_ + "_input"] = x_test[i];
        
        graph_->forward(feeds);
        auto output = graph_->get_value(output_node_->name());
        auto target = y_test[i];
        
        auto diff = sub(*output, *target);
        auto squared = mul(diff, diff);
        auto loss_tensor = sum(squared);
        total_loss += loss_tensor.data()[0] / output->size();
    }
    
    return total_loss / x_test.size();
}

void Model::save(const std::string& path) {
    graph_->save(path);
}

std::shared_ptr<Model> Model::load(const std::string& path) {
    auto model = std::make_shared<Model>("loaded_model");
    model->graph_ = Graph::load(path);
    model->compiled_ = true;
    return model;
}

void Model::compute_loss(std::shared_ptr<Tensor> y_pred, std::shared_ptr<Tensor> y_true) {
    (void)y_pred;
    (void)y_true;
    // Loss computation is handled in fit()
}

void Model::update_parameters() {
    // Simple SGD update
    for (auto& layer : layers_) {
        auto params = layer->parameters();
        for (auto& param : params) {
            if (param->grad()) {
                auto grad_data = param->grad()->data();
                auto param_data = param->data();
                for (size_t i = 0; i < param->size(); i++) {
                    param_data[i] -= learning_rate_ * grad_data[i];
                }
            }
        }
    }
}

// === Sequential ===
Sequential::Sequential(const std::string& name) : Model(name) {}

// === MLP ===
MLP::MLP(const std::vector<int>& hidden_sizes, 
         const std::string& activation,
         const std::string& name) : Model(name) {
    
    for (size_t i = 0; i < hidden_sizes.size() - 1; i++) {
        int input_size = hidden_sizes[i];
        int output_size = hidden_sizes[i + 1];
        
        std::string layer_name = name + "_dense_" + std::to_string(i);
        add_layer(std::make_shared<DenseLayer>(input_size, output_size, activation, layer_name));
    }
}

// === CNN ===
CNN::CNN(const std::vector<int>& conv_channels, 
         const std::vector<int>& dense_sizes,
         const std::string& activation,
         const std::string& name) : Model(name) {
    
    // Add convolutional layers
    for (size_t i = 0; i < conv_channels.size() - 1; i++) {
        int in_ch = conv_channels[i];
        int out_ch = conv_channels[i + 1];
        
        std::string conv_name = name + "_conv_" + std::to_string(i);
        add_layer(std::make_shared<Conv2DLayer>(in_ch, out_ch, 3, 1, 1, activation, conv_name));
        
        std::string pool_name = name + "_pool_" + std::to_string(i);
        add_layer(std::make_shared<MaxPool2DLayer>(2, 2, pool_name));
    }
    
    // Flatten
    add_layer(std::make_shared<FlattenLayer>(name + "_flatten"));
    
    // Add dense layers
    for (size_t i = 0; i < dense_sizes.size() - 1; i++) {
        int input_size = dense_sizes[i];
        int output_size = dense_sizes[i + 1];
        
        std::string dense_name = name + "_dense_" + std::to_string(i);
        add_layer(std::make_shared<DenseLayer>(input_size, output_size, activation, dense_name));
    }
}

} // namespace adaicpp
