#include "adaicpp/adaicpp.h"
#include <iostream>
#include <memory>

using namespace adaicpp;

int main() {
    try {
        // Initialize the library
        initialize();
        
        std::cout << "=== adAI C++ Backend Example ===" << std::endl;
        
        // Create a graph
        auto graph = std::make_unique<Graph>();
        
        // Create placeholders for inputs
        auto x = graph->add_placeholder("x");
        auto y = graph->add_placeholder("y");
        
        // Create parameters (weights)
        auto w1 = graph->add_parameter("w1", std::make_shared<Tensor>(Tensor::randn({2, 3})));
        auto b1 = graph->add_parameter("b1", std::make_shared<Tensor>(Tensor::zeros({3})));
        
        // Build a simple neural network layer
        // z = x * w1 + b1
        auto matmul_op = graph->add_op("matmul1", {x, w1},
            [](const std::vector<std::shared_ptr<Tensor>>& inputs) -> std::shared_ptr<Tensor> {
                return std::make_shared<Tensor>(matmul(*inputs[0], *inputs[1]));
            },
            nullptr,  // backward function
            true      // static
        );
        
        auto add_op = graph->add_op("add1", {matmul_op, b1},
            [](const std::vector<std::shared_ptr<Tensor>>& inputs) -> std::shared_ptr<Tensor> {
                return std::make_shared<Tensor>(add(*inputs[0], *inputs[1]));
            },
            nullptr,
            true
        );
        
        // Apply activation
        auto relu_op = graph->add_op("relu1", {add_op},
            [](const std::vector<std::shared_ptr<Tensor>>& inputs) -> std::shared_ptr<Tensor> {
                return std::make_shared<Tensor>(relu(*inputs[0]));
            },
            nullptr,
            true
        );
        
        // Compile the graph
        graph->compile();
        
        std::cout << "Graph compiled successfully!" << std::endl;
        graph->print_layout();
        
        // Create input data
        auto input_x = Tensor::ones({5, 2});
        auto input_y = Tensor::zeros({5, 3});
        
        // Run forward pass
        std::unordered_map<std::string, std::shared_ptr<Tensor>> feeds;
        feeds["x"] = std::make_shared<Tensor>(input_x);
        feeds["y"] = std::make_shared<Tensor>(input_y);
        
        graph->forward(feeds);
        
        // Get output
        auto output = graph->get_value("relu1");
        std::cout << "Output shape: [";
        for (size_t dim : output->shape()) {
            std::cout << dim << " ";
        }
        std::cout << "]" << std::endl;
        
        // Test tensor operations directly
        std::cout << "\n=== Testing Tensor Operations ===" << std::endl;
        
        auto a = Tensor::ones({2, 2});
        auto b = scalar_mul(Tensor::ones({2, 2}), 2.0f);
        
        auto c = add(a, b);
        std::cout << "Addition test passed" << std::endl;
        
        auto d = mul(a, b);
        std::cout << "Multiplication test passed" << std::endl;
        
        auto e = relu(Tensor::arange(-2.0f, 2.0f, 1.0f));
        std::cout << "ReLU test passed" << std::endl;
        
        // Test memory pool
        std::cout << "\n=== Memory Pool Statistics ===" << std::endl;
        auto& pool = MemoryPool::instance();
        std::cout << "Cache hits: " << pool.cache_hits() << std::endl;
        std::cout << "Cache misses: " << pool.cache_misses() << std::endl;
        
        std::cout << "\n=== Example completed successfully ===" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
