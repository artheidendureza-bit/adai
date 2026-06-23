#pragma once

#include "adaicpp/graph.h"
#include "adaicpp/tensor.h"
#include "adaicpp/device.h"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>

namespace adaicpp {

// Forward declarations
class Layer;
class Model;

// Layer base class
class Layer {
public:
    virtual ~Layer() = default;
    
    virtual std::string name() const = 0;
    virtual std::vector<std::shared_ptr<Tensor>> parameters() const = 0;
    virtual void set_parameters(const std::vector<std::shared_ptr<Tensor>>& params) = 0;
    
    virtual std::shared_ptr<Node> build(std::shared_ptr<Graph> graph, 
                                       std::shared_ptr<Node> input) = 0;
};

// Dense (fully connected) layer
class DenseLayer : public Layer {
public:
    DenseLayer(int input_size, int output_size, 
               const std::string& activation = "relu",
               const std::string& name = "dense");
    
    std::string name() const override { return name_; }
    std::vector<std::shared_ptr<Tensor>> parameters() const override;
    void set_parameters(const std::vector<std::shared_ptr<Tensor>>& params) override;
    
    std::shared_ptr<Node> build(std::shared_ptr<Graph> graph, 
                               std::shared_ptr<Node> input) override;
    
private:
    int input_size_;
    int output_size_;
    std::string activation_;
    std::string name_;
    std::shared_ptr<Tensor> weights_;
    std::shared_ptr<Tensor> bias_;
};

// Conv2D layer
class Conv2DLayer : public Layer {
public:
    Conv2DLayer(int in_channels, int out_channels, 
               int kernel_size, int stride = 1, int padding = 0,
               const std::string& activation = "relu",
               const std::string& name = "conv2d");
    
    std::string name() const override { return name_; }
    std::vector<std::shared_ptr<Tensor>> parameters() const override;
    void set_parameters(const std::vector<std::shared_ptr<Tensor>>& params) override;
    
    std::shared_ptr<Node> build(std::shared_ptr<Graph> graph, 
                               std::shared_ptr<Node> input) override;
    
private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    std::string activation_;
    std::string name_;
    std::shared_ptr<Tensor> weights_;
    std::shared_ptr<Tensor> bias_;
};

// MaxPool2D layer
class MaxPool2DLayer : public Layer {
public:
    MaxPool2DLayer(int pool_size, int stride = -1, 
                  const std::string& name = "maxpool2d");
    
    std::string name() const override { return name_; }
    std::vector<std::shared_ptr<Tensor>> parameters() const override { return {}; }
    void set_parameters(const std::vector<std::shared_ptr<Tensor>>& params) override { (void)params; }
    
    std::shared_ptr<Node> build(std::shared_ptr<Graph> graph, 
                               std::shared_ptr<Node> input) override;
    
private:
    int pool_size_;
    int stride_;
    std::string name_;
};

// Flatten layer
class FlattenLayer : public Layer {
public:
    FlattenLayer(const std::string& name = "flatten");
    
    std::string name() const override { return name_; }
    std::vector<std::shared_ptr<Tensor>> parameters() const override { return {}; }
    void set_parameters(const std::vector<std::shared_ptr<Tensor>>& params) override { (void)params; }
    
    std::shared_ptr<Node> build(std::shared_ptr<Graph> graph, 
                               std::shared_ptr<Node> input) override;
    
private:
    std::string name_;
};

// Training callback
class TrainingCallback {
public:
    virtual ~TrainingCallback() = default;
    virtual void on_epoch_begin(int epoch) {}
    virtual void on_epoch_end(int epoch, float loss, float metric) {}
    virtual void on_batch_begin(int batch, int total_batches) {}
    virtual void on_batch_end(int batch, float loss) {}
};

// Model class with training API
class Model {
public:
    Model(const std::string& name = "model");
    
    // Add layers
    void add_layer(std::shared_ptr<Layer> layer);
    
    // Build the computational graph
    void build(const std::vector<size_t>& input_shape);
    
    // Compile the model
    void compile(const std::string& optimizer = "adam", 
                float learning_rate = 0.001f,
                const std::string& loss = "mse");
    
    // Training
    void fit(
        const std::vector<std::shared_ptr<Tensor>>& x_train,
        const std::vector<std::shared_ptr<Tensor>>& y_train,
        int epochs = 10,
        int batch_size = 32,
        const std::vector<std::shared_ptr<Tensor>>& x_val = {},
        const std::vector<std::shared_ptr<Tensor>>& y_val = {},
        std::vector<std::shared_ptr<TrainingCallback>> callbacks = {}
    );
    
    // Prediction
    std::shared_ptr<Tensor> predict(std::shared_ptr<Tensor> x);
    
    // Evaluation
    float evaluate(const std::vector<std::shared_ptr<Tensor>>& x_test,
                  const std::vector<std::shared_ptr<Tensor>>& y_test);
    
    // Save/Load
    void save(const std::string& path);
    static std::shared_ptr<Model> load(const std::string& path);
    
    // Getters
    std::shared_ptr<Graph> graph() const { return graph_; }
    std::string name() const { return name_; }
    
private:
    std::string name_;
    std::vector<std::shared_ptr<Layer>> layers_;
    std::shared_ptr<Graph> graph_;
    std::shared_ptr<Node> input_node_;
    std::shared_ptr<Node> output_node_;
    std::vector<size_t> input_shape_;
    
    std::string optimizer_;
    float learning_rate_;
    std::string loss_;
    
    bool built_;
    bool compiled_;
    
    // Training history
    std::vector<float> loss_history_;
    std::vector<float> val_loss_history_;
    
    // Helper methods
    void initialize_parameters();
    void compute_loss(std::shared_ptr<Tensor> y_pred, std::shared_ptr<Tensor> y_true);
    void update_parameters();
};

// Preset architectures
class Sequential : public Model {
public:
    Sequential(const std::string& name = "sequential");
};

class MLP : public Model {
public:
    MLP(const std::vector<int>& hidden_sizes, 
        const std::string& activation = "relu",
        const std::string& name = "mlp");
};

class CNN : public Model {
public:
    CNN(const std::vector<int>& conv_channels, 
        const std::vector<int>& dense_sizes,
        const std::string& activation = "relu",
        const std::string& name = "cnn");
};

} // namespace adaicpp
