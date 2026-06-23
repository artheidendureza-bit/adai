# Building Python Bindings for adAI C++ Backend

This guide explains how to build the Python bindings for the adAI C++ backend.

## Prerequisites

- CMake 3.15 or higher
- C++17 compatible compiler (GCC, Clang, MSVC)
- Python 3.6 or higher
- pybind11
- (Optional) CUDA Toolkit 11.0 or higher

## Installing pybind11

### Using pip
```bash
pip install pybind11
```

### Using conda
```bash
conda install -c conda-forge pybind11
```

### Building from source
```bash
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build && cd build
cmake ..
make install
```

## Building the Python Module

### Method 1: Using CMake (Recommended)

```bash
cd cpp_backend
mkdir build
cd build

# Configure with Python bindings enabled
cmake .. -DADAICPP_BUILD_PYTHON=ON

# Build
cmake --build . --config Release

# The compiled module will be copied to ../src/adai/
```

### Method 2: Using setup.py

```bash
cd cpp_backend/python_bindings
pip install -e .
```

### Method 3: Using pip directly

```bash
cd cpp_backend
pip install -e .
```

## Build Options

- `-DADAICPP_BUILD_PYTHON=ON`: Enable Python bindings (default: ON)
- `-DADAICPP_ENABLE_CUDA=ON`: Enable CUDA support (default: OFF)
- `-DADAICPP_BUILD_TESTS=ON`: Build tests (default: ON)

## Building with CUDA Support

```bash
cmake .. -DADAICPP_BUILD_PYTHON=ON -DADAICPP_ENABLE_CUDA=ON
cmake --build . --config Release
```

## Troubleshooting

### pybind11 not found
If CMake cannot find pybind11, you can specify its location:
```bash
cmake .. -DPYBIND11_DIR=/path/to/pybind11
```

### Python not found
If CMake cannot find Python, specify the Python executable:
```bash
cmake .. -DPython3_EXECUTABLE=/path/to/python
```

### Module not found after build
Ensure the compiled module is in the correct location:
```bash
# The module should be at: ../src/adai/adaicpp.pyd (Windows) or adaicpp.so (Linux/Mac)
```

Add the adAI source directory to your Python path:
```bash
export PYTHONPATH=/path/to/adAI/src:$PYTHONPATH
```

## Using the Python Bindings

```python
import adai
from adai.backend import set_backend, Backend, is_cpp_available

# Check if C++ backend is available
if is_cpp_available():
    print("C++ backend is available!")
    
    # Set backend to C++
    set_backend(Backend.CPP)
    
    # Use C++ tensor operations
    from adai.cpp_tensor import CppTensor, add, mul, matmul, relu
    
    # Create tensors
    a = CppTensor([2, 3])
    b = CppTensor([2, 3])
    
    # Perform operations
    c = add(a, b)
    d = relu(c)
    
    print(f"Result shape: {d.shape}")
else:
    print("C++ backend not available, using Python backend")
```

## Architecture

The Python bindings follow this architecture:

```
Python adAI Frontend (User API)
    ↓
Python Bindings (pybind11)
    ↓
C++ Backend (adaicpp library)
    ↓
CUDA Kernels (GPU acceleration, optional)
```

### Components

1. **Python Frontend**: User-facing API in `src/adai/`
   - Model definitions
   - Graph building
   - High-level operations

2. **Python Bindings**: `cpp_backend/python_bindings/`
   - `tensor_bindings.cpp`: Tensor class bindings
   - `device_bindings.cpp`: Device abstraction bindings
   - `node_bindings.cpp`: Node hierarchy bindings
   - `graph_bindings.cpp`: Graph class bindings
   - `op_registry_bindings.cpp`: Operations registry bindings
   - `module.cpp`: Main pybind11 module

3. **C++ Backend**: `cpp_backend/src/`
   - Tensor operations
   - Graph execution
   - Autograd
   - Memory management

4. **CUDA Kernels**: `cpp_backend/src/cuda/` (optional)
   - GPU-accelerated operations
   - matmul, conv2d, etc.

## Performance Considerations

- The C++ backend provides significant performance improvements for:
  - Large tensor operations
  - Matrix multiplication
  - Convolution operations
  - Batch processing

- Memory pooling reduces allocation overhead
- Static node caching for repeated computations

## Development

### Adding New Python Bindings

1. Add the binding function in the appropriate `*_bindings.cpp` file
2. Call the binding function in `module.cpp`
3. Rebuild the Python module

### Debugging

To enable debug symbols:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

To see C++ output in Python:
```bash
# The C++ backend prints to stdout/stderr
# Run Python with output visible
python your_script.py
```
