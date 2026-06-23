# adAI C++ Backend

A high-performance C++ backend for the adAI machine learning library, featuring a hybrid static/dynamic computational graph with automatic differentiation.

## Features

- **Hybrid Static/Dynamic Execution**: Nodes can be compiled as static (cached) or dynamic (eager evaluation)
- **Automatic Differentiation**: Reverse-mode autograd with gradient tracking
- **Device Abstraction**: Unified interface for CPU and CUDA (when enabled)
- **Memory Pooling**: Efficient buffer reuse to minimize allocation overhead
- **Graph Optimization**: Dead code elimination and constant folding
- **Tensor Operations**: Element-wise operations, matrix multiplication, activations
- **Serialization**: Graph schema and parameter serialization to JSON

## Architecture

### Core Components

- **Tensor**: Multi-dimensional arrays with shape, strides, and device abstraction
- **Device**: Abstract interface for CPU and CUDA devices
- **MemoryPool**: Buffer pooling for efficient memory management
- **Node**: Base class for computational graph nodes (Placeholder, Parameter, Operation)
- **Graph**: Manages the computational graph with topological sorting and compilation
- **OpRegistry**: Registry for operations with forward/backward functions

### Node Types

- **PlaceholderNode**: External inputs (always dynamic)
- **ParameterNode**: Model weights/biases (static unless updated)
- **OpNode**: Operations with forward/backward functions

## Building

### Prerequisites

- CMake 3.15 or higher
- C++17 compatible compiler (GCC, Clang, MSVC)
- (Optional) CUDA Toolkit 11.0 or higher for CUDA support

### Build Commands

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake (CPU only)
cmake ..

# Configure with CUDA support
cmake .. -DADAICPP_ENABLE_CUDA=ON

# Build
cmake --build .

# Run tests
ctest

# Run example
./basic_example
```

## Usage Example

```cpp
#include "adaicpp/adaicpp.h"

using namespace adaicpp;

int main() {
    // Initialize library
    initialize();
    
    // Create graph
    auto graph = std::make_unique<Graph>();
    
    // Add placeholders
    auto x = graph->add_placeholder("x");
    
    // Add parameters
    auto w = graph->add_parameter("w", Tensor::randn({2, 3}));
    auto b = graph->add_parameter("b", Tensor::zeros({3}));
    
    // Build operations
    auto matmul = graph->add_op("matmul", {x, w},
        [](const auto& inputs) { return matmul(*inputs[0], *inputs[1]); },
        nullptr, true);
    
    auto add = graph->add_op("add", {matmul, b},
        [](const auto& inputs) { return add(*inputs[0], *inputs[1]); },
        nullptr, true);
    
    auto relu = graph->add_op("relu", {add},
        [](const auto& inputs) { return relu(*inputs[0]); },
        nullptr, true);
    
    // Compile graph
    graph->compile();
    
    // Run forward pass
    auto input = Tensor::ones({5, 2});
    std::unordered_map<std::string, std::shared_ptr<Tensor>> feeds;
    feeds["x"] = std::make_shared<Tensor>(input);
    
    graph->forward(feeds);
    
    // Get output
    auto output = graph->get_value("relu");
    
    // Run backward pass
    graph->backward("relu");
    
    // Get gradients
    auto w_grad = graph->get_grad("w");
    
    return 0;
}
```

## Project Structure

```
cpp_backend/
├── CMakeLists.txt
├── README.md
├── include/
│   └── adaicpp/
│       ├── adaicpp.h          # Main header
│       ├── device.h           # Device abstraction
│       ├── tensor.h           # Tensor class
│       ├── memory_pool.h      # Memory pooling
│       ├── node.h             # Node hierarchy
│       ├── graph.h            # Graph management
│       └── op_registry.h      # Operation registry
├── src/
│   ├── core/
│   │   ├── device.cpp
│   │   ├── tensor.cpp
│   │   └── memory_pool.cpp
│   ├── graph/
│   │   ├── node.cpp
│   │   └── graph.cpp
│   └── ops/
│       ├── op_registry.cpp
│       └── operations.cpp
├── tests/
│   ├── test_tensor.cpp
│   └── test_graph.cpp
└── examples/
    └── basic_example.cpp
```

## Supported Operations

### Element-wise
- `add`: Addition
- `sub`: Subtraction
- `mul`: Multiplication
- `div`: Division

### Activations
- `relu`: Rectified Linear Unit
- `sigmoid`: Sigmoid activation
- `tanh`: Hyperbolic tangent

### Matrix Operations
- `matmul`: Matrix multiplication
- `transpose`: Tensor transposition

### Reductions
- `sum`: Sum reduction
- `mean`: Mean reduction

## Design Decisions

### Static vs Dynamic Execution

- **Static nodes**: Cache results, recompute only when inputs change (dirty flag)
- **Dynamic nodes**: Always recompute (eager evaluation like PyTorch)
- **Hybrid graphs**: Static subgraphs can exist within dynamic graphs

### Memory Management

- Memory pooling reduces allocation overhead
- Buffer reuse when shapes match
- Device-aware memory management for CPU/CUDA

### Autograd

- Reverse-mode automatic differentiation
- Gradient accumulation for parameters
- Computation graph traversal for backward pass

## Performance Considerations

- Zero-cost abstractions where possible
- Template-based device dispatch
- Memory pooling to minimize allocations
- Static node caching for repeated computations
- Graph optimization (DCE, constant folding)

## Future Work

- [ ] Convolution operations (2D)
- [ ] More activation functions
- [ ] Optimizer implementations (SGD, Adam, etc.)
- [ ] Layer abstractions (Dense, Conv2D, etc.)
- [ ] CUDA kernel implementations
- [ ] Multi-GPU support
- [ ] Distributed training support

## License

MIT License - See parent project LICENSE file for details.
