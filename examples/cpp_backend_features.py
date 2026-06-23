"""
Examples demonstrating advanced C++ backend features:
- CUDA operations
- Preset architectures
- Model training with fit()
- ONNX export
- Mixed precision training
"""

import numpy as np
import sys

# Try to import C++ backend
try:
    import adAI
    CPP_BACKEND_AVAILABLE = adAI.is_cpp_backend_available()
except ImportError:
    CPP_BACKEND_AVAILABLE = False
    print("C++ backend not available")

if CPP_BACKEND_AVAILABLE:
    import adaicpp_py as _cpp


def example_cuda_operations():
    """Demonstrate CUDA-accelerated operations"""
    print("\n=== CUDA Operations Example ===")
    
    if not CPP_BACKEND_AVAILABLE:
        print("C++ backend not available, skipping")
        return
    
    try:
        # Create tensors
        a = _cpp.Tensor([1024, 1024])
        b = _cpp.Tensor([1024, 1024])
        
        # Fill with random values
        a_data = np.random.randn(1024, 1024).astype(np.float32)
        b_data = np.random.randn(1024, 1024).astype(np.float32)
        
        # Convert to C++ tensors
        a_cpp = _cpp.numpy_to_tensor(a_data)
        b_cpp = _cpp.numpy_to_tensor(b_data)
        
        # Perform CUDA matmul
        c_cpp = _cpp.matmul(a_cpp, b_cpp)
        
        print("✓ CUDA matrix multiplication completed")
        print(f"  Result shape: {c_cpp.shape()}")
        
    except Exception as e:
        print(f"✗ CUDA operations failed: {e}")


def example_preset_architectures():
    """Demonstrate preset architecture classes"""
    print("\n=== Preset Architectures Example ===")
    
    if not CPP_BACKEND_AVAILABLE:
        print("C++ backend not available, skipping")
        return
    
    try:
        # Create MLP
        mlp = _cpp.MLP([784, 256, 128, 10], "relu", "example_mlp")
        print("✓ Created MLP architecture")
        print(f"  Input: 784, Hidden: [256, 128], Output: 10")
        
        # Create CNN
        cnn = _cpp.CNN([3, 32, 64], [1024, 256, 10], "relu", "example_cnn")
        print("✓ Created CNN architecture")
        print(f"  Conv channels: [3, 32, 64], Dense: [1024, 256, 10]")
        
    except Exception as e:
        print(f"✗ Preset architectures failed: {e}")


def example_model_training():
    """Demonstrate model.fit() training API"""
    print("\n=== Model Training Example ===")
    
    if not CPP_BACKEND_AVAILABLE:
        print("C++ backend not available, skipping")
        return
    
    try:
        # Create simple MLP
        model = _cpp.MLP([4, 8, 2], "relu", "training_example")
        
        # Build model
        model.build([4])
        print("✓ Model built")
        
        # Compile model
        model.compile("adam", 0.01f, "mse")
        print("✓ Model compiled")
        
        # Create dummy training data
        x_train = [_cpp.numpy_to_tensor(np.random.randn(4).astype(np.float32)) for _ in range(100)]
        y_train = [_cpp.numpy_to_tensor(np.random.randn(2).astype(np.float32)) for _ in range(100)]
        
        # Train model
        print("Training model...")
        model.fit(x_train, y_train, epochs=5, batch_size=32)
        print("✓ Training completed")
        
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        import traceback
        traceback.print_exc()


def example_onnx_export():
    """Demonstrate ONNX export functionality"""
    print("\n=== ONNX Export Example ===")
    
    if not CPP_BACKEND_AVAILABLE:
        print("C++ backend not available, skipping")
        return
    
    try:
        # Create a simple graph
        graph = _cpp.Graph()
        
        # Add nodes
        x = graph.add_placeholder("x")
        w = graph.add_parameter("w", _cpp.numpy_to_tensor(np.ones((4, 2), dtype=np.float32)))
        b = graph.add_parameter("b", _cpp.numpy_to_tensor(np.zeros((2,), dtype=np.float32)))
        
        # Build operations
        matmul = graph.add_op("matmul", "matmul", [x, w])
        add = graph.add_op("add", "add", [matmul, b])
        relu = graph.add_op("relu", "relu", [add])
        
        # Compile
        graph.compile()
        
        # Export to ONNX
        onnx_str = _cpp.ONNXExporter.export_to_onnx(graph, "example_model")
        print("✓ ONNX export completed")
        print(f"  ONNX proto length: {len(onnx_str)} characters")
        
        # Save to file
        _cpp.ONNXExporter.save_onnx(graph, "example_model.onnx", "example_model")
        print("✓ ONNX model saved to file")
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()


def example_mixed_precision():
    """Demonstrate mixed precision training"""
    print("\n=== Mixed Precision Training Example ===")
    
    if not CPP_BACKEND_AVAILABLE:
        print("C++ backend not available, skipping")
        return
    
    try:
        # Check if mixed precision is supported
        device = _cpp.DeviceManager.instance().get_cpu_device()
        supported = _cpp.supports_mixed_precision(device)
        
        if not supported:
            print("⊘ Mixed precision not supported on CPU (requires CUDA)")
            return
        
        # Create model
        model = _cpp.MLP([4, 8, 2], "relu", "mixed_precision_example")
        model.build([4])
        model.compile("adam", 0.01f, "mse")
        
        # Create mixed precision trainer
        trainer = _cpp.MixedPrecisionTrainer(model, loss_scale=1.0f)
        print("✓ Mixed precision trainer created")
        print(f"  Initial loss scale: {trainer.get_loss_scale()}")
        
        # Convert to mixed precision
        trainer.convert_to_mixed_precision()
        print("✓ Model converted to mixed precision")
        
        # Simulate training step
        x_batch = _cpp.numpy_to_tensor(np.random.randn(4).astype(np.float32))
        y_batch = _cpp.numpy_to_tensor(np.random.randn(2).astype(np.float32))
        
        # Note: Full training step would require complete implementation
        print("⊘ Training step placeholder (requires full implementation)")
        
    except Exception as e:
        print(f"✗ Mixed precision training failed: {e}")
        import traceback
        traceback.print_exc()


def example_convolution():
    """Demonstrate 2D convolution operation"""
    print("\n=== Convolution Operation Example ===")
    
    if not CPP_BACKEND_AVAILABLE:
        print("C++ backend not available, skipping")
        return
    
    try:
        # Create input tensor [batch, channels, height, width]
        input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
        input_cpp = _cpp.numpy_to_tensor(input_data)
        
        # Create kernel tensor [out_channels, in_channels, height, width]
        kernel_data = np.random.randn(16, 3, 3, 3).astype(np.float32)
        kernel_cpp = _cpp.numpy_to_tensor(kernel_data)
        
        # Perform convolution
        output_cpp = _cpp.conv2d(input_cpp, kernel_cpp, stride_h=1, stride_w=1, padding_h=1, padding_w=1)
        
        print("✓ 2D convolution completed")
        print(f"  Input shape: {input_cpp.shape()}")
        print(f"  Kernel shape: {kernel_cpp.shape()}")
        print(f"  Output shape: {output_cpp.shape()}")
        
    except Exception as e:
        print(f"✗ Convolution failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all examples"""
    print("=" * 60)
    print("adAI C++ Backend Advanced Features Examples")
    print("=" * 60)
    
    if not CPP_BACKEND_AVAILABLE:
        print("\nC++ backend not available. Build it with:")
        print("  python cpp_backend/build_ext.py")
        return
    
    examples = [
        example_cuda_operations,
        example_preset_architectures,
        example_model_training,
        example_onnx_export,
        example_mixed_precision,
        example_convolution,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Example failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
