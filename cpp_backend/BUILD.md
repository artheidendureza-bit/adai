# Building the adAI C++ Backend

The adAI C++ backend provides high-performance computational graph execution with hybrid static/dynamic optimization. This guide explains how to build and integrate it with the Python adAI package.

## Prerequisites

- **CMake** >= 3.15
- **C++17 compatible compiler** (GCC, Clang, MSVC)
- **Python** >= 3.8
- **pybind11** >= 2.10.0
- **Ninja** (recommended for faster builds)

### Windows (Visual Studio)
```bash
# Install using chocolatey or download from Microsoft
choco install cmake ninja
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install cmake ninja-build python3-dev pybind11-dev
```

### macOS
```bash
brew install cmake ninja pybind11
```

## Building the C++ Extension

### Option 1: Using the build script (Recommended)

```bash
# From the adAI root directory
python cpp_backend/build_ext.py
```

This will:
1. Configure the C++ backend with CMake
2. Build the Python extension
3. Copy the compiled extension to `src/adAI/`

### Option 2: Manual build with CMake

```bash
cd cpp_backend
mkdir build
cd build
cmake .. -DADAICPP_BUILD_PYTHON=ON -DADAICPP_BUILD_TESTS=OFF
cmake --build . --config Release
```

Then copy the built extension to the Python source directory:
```bash
# Windows
copy Release\adaicpp_py.pyd ..\..\src\adAI\

# Linux/macOS
cp adaicpp_py*.so ../../src/adAI/
```

## Building with CUDA Support (Optional)

To enable CUDA support, add the CUDA flag:

```bash
python cpp_backend/build_ext.py
# Or manually:
cmake .. -DADAICPP_BUILD_PYTHON=ON -DADAICPP_ENABLE_CUDA=ON
```

**Note:** This requires:
- NVIDIA CUDA Toolkit
- Compatible NVIDIA GPU
- CUDA-capable compiler

## Building Tests

To build and run the C++ unit tests:

```bash
cd cpp_backend
mkdir build
cd build
cmake .. -DADAICPP_BUILD_TESTS=ON
cmake --build .
ctest
```

## Verifying the Installation

Run the integration test:

```bash
python test_cpp_integration.py
```

This will verify that:
- The C++ extension can be imported
- Graph creation works
- Basic operations function correctly
- Tensor operations are available

## Using the C++ Backend in Python

```python
import adAI
import numpy as np

# Check if C++ backend is available
if adAI.is_cpp_backend_available():
    # Create a C++ graph
    graph = adAI.create_graph()
    
    # Add placeholders
    x = graph.add_placeholder("x")
    y = graph.add_placeholder("y")
    
    # Add parameters
    w = graph.add_parameter("w", np.ones((2, 3), dtype=np.float32))
    
    # Add operations
    add_op = graph.add_op("add", "add", [x, y])
    
    # Compile
    graph.compile()
    
    # Forward pass
    feeds = {
        "x": np.ones((2, 3), dtype=np.float32),
        "y": np.ones((2, 3), dtype=np.float32) * 2.0
    }
    results = graph.forward(feeds)
    
    print("Result:", results['add'])
else:
    print("C++ backend not available, using Python implementation")
```

## Troubleshooting

### Import Error: "No module named 'adaicpp_py'"

- Ensure you've built the C++ extension using one of the methods above
- Check that the extension file exists in `src/adAI/`
- On Windows, look for `adaicpp_py.pyd`
- On Linux/macOS, look for `adaicpp_py*.so`

### CMake Configuration Errors

- Ensure CMake >= 3.15 is installed
- Verify that your C++ compiler supports C++17
- Check that Python development headers are installed

### pybind11 Not Found

- Install pybind11: `pip install pybind11`
- Or install system package: `sudo apt-get install pybind11-dev` (Linux)

### Build Errors on Windows

- Ensure you have Visual Studio with C++ development tools
- Use the "Developer Command Prompt for VS" to run build commands
- Make sure CMake can find your Visual Studio installation

## Performance Considerations

The C++ backend provides significant performance benefits for:

- **Large tensor operations**: Matrix multiplication, element-wise operations
- **Complex graphs**: Deep networks with many layers
- **Static subgraphs**: Repeated computations benefit from caching
- **Memory-intensive operations**: Memory pooling reduces allocation overhead

For small graphs or infrequent computations, the Python implementation may be sufficient and easier to debug.

## Architecture

The C++ backend includes:

- **Tensor**: Multi-dimensional arrays with device abstraction
- **Device**: CPU and CUDA device management
- **MemoryPool**: Efficient buffer reuse
- **Graph**: Hybrid static/dynamic computational graph
- **OpRegistry**: Operation registry with forward/backward functions
- **Autograd**: Reverse-mode automatic differentiation

The Python wrapper (`src/adAI/cpp_backend.py`) provides a Pythonic interface while maintaining C++ performance.
