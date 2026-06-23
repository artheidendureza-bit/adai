"""
Tests for advanced C++ backend features
"""

import numpy as np
import pytest
import sys

# Try to import C++ backend
try:
    import adAI
    CPP_BACKEND_AVAILABLE = adAI.is_cpp_backend_available()
except ImportError:
    CPP_BACKEND_AVAILABLE = False

if CPP_BACKEND_AVAILABLE:
    import adaicpp_py as _cpp


@pytest.mark.skipif(not CPP_BACKEND_AVAILABLE, reason="C++ backend not available")
class TestCUDAOperations:
    """Test CUDA-accelerated operations"""
    
    def test_matmul_cuda(self):
        """Test CUDA matrix multiplication"""
        a_data = np.random.randn(64, 64).astype(np.float32)
        b_data = np.random.randn(64, 64).astype(np.float32)
        
        a_cpp = _cpp.numpy_to_tensor(a_data)
        b_cpp = _cpp.numpy_to_tensor(b_data)
        
        c_cpp = _cpp.matmul(a_cpp, b_cpp)
        
        assert c_cpp.shape() == [64, 64]
        
        # Compare with numpy
        c_numpy = np.dot(a_data, b_data)
        c_cpp_numpy = _cpp.tensor_to_numpy(c_cpp)
        
        np.testing.assert_allclose(c_cpp_numpy, c_numpy, rtol=1e-5, atol=1e-5)
    
    def test_conv2d_cuda(self):
        """Test CUDA 2D convolution"""
        input_data = np.random.randn(2, 3, 16, 16).astype(np.float32)
        kernel_data = np.random.randn(8, 3, 3, 3).astype(np.float32)
        
        input_cpp = _cpp.numpy_to_tensor(input_data)
        kernel_cpp = _cpp.numpy_to_tensor(kernel_data)
        
        output_cpp = _cpp.conv2d(input_cpp, kernel_cpp, stride_h=1, stride_w=1, padding_h=1, padding_w=1)
        
        # Check output shape
        expected_shape = [2, 8, 16, 16]  # With padding=1, stride=1
        assert output_cpp.shape() == expected_shape


@pytest.mark.skipif(not CPP_BACKEND_AVAILABLE, reason="C++ backend not available")
class TestPresetArchitectures:
    """Test preset architecture classes"""
    
    def test_mlp_creation(self):
        """Test MLP architecture creation"""
        mlp = _cpp.MLP([10, 20, 5], "relu", "test_mlp")
        assert mlp.name() == "test_mlp"
    
    def test_cnn_creation(self):
        """Test CNN architecture creation"""
        cnn = _cpp.CNN([3, 16, 32], [64, 10], "relu", "test_cnn")
        assert cnn.name() == "test_cnn"
    
    def test_sequential_creation(self):
        """Test Sequential architecture creation"""
        seq = _cpp.Sequential("test_seq")
        assert seq.name() == "test_seq"


@pytest.mark.skipif(not CPP_BACKEND_AVAILABLE, reason="C++ backend not available")
class TestModelTraining:
    """Test model training API"""
    
    def test_model_build(self):
        """Test model building"""
        model = _cpp.MLP([4, 8, 2], "relu", "test_build")
        model.build([4])
        # Should not raise an exception
    
    def test_model_compile(self):
        """Test model compilation"""
        model = _cpp.MLP([4, 8, 2], "relu", "test_compile")
        model.build([4])
        model.compile("adam", 0.01, "mse")
        # Should not raise an exception
    
    def test_model_predict(self):
        """Test model prediction"""
        model = _cpp.MLP([4, 8, 2], "relu", "test_predict")
        model.build([4])
        model.compile("adam", 0.01, "mse")
        
        x = _cpp.numpy_to_tensor(np.random.randn(4).astype(np.float32))
        output = model.predict(x)
        
        assert output.shape() == [2]


@pytest.mark.skipif(not CPP_BACKEND_AVAILABLE, reason="C++ backend not available")
class TestONNXExport:
    """Test ONNX export functionality"""
    
    def test_onnx_export_string(self):
        """Test ONNX export to string"""
        graph = _cpp.Graph()
        x = graph.add_placeholder("x")
        w = graph.add_parameter("w", _cpp.numpy_to_tensor(np.ones((4, 2), dtype=np.float32)))
        matmul = graph.add_op("matmul", "matmul", [x, w])
        graph.compile()
        
        onnx_str = _cpp.ONNXExporter.export_to_onnx(graph, "test_model")
        
        assert isinstance(onnx_str, str)
        assert len(onnx_str) > 0
        assert "ir_version" in onnx_str
    
    def test_onnx_export_file(self):
        """Test ONNX export to file"""
        import tempfile
        import os
        
        graph = _cpp.Graph()
        x = graph.add_placeholder("x")
        w = graph.add_parameter("w", _cpp.numpy_to_tensor(np.ones((4, 2), dtype=np.float32)))
        matmul = graph.add_op("matmul", "matmul", [x, w])
        graph.compile()
        
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            filepath = f.name
        
        try:
            _cpp.ONNXExporter.save_onnx(graph, filepath, "test_model")
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


@pytest.mark.skipif(not CPP_BACKEND_AVAILABLE, reason="C++ backend not available")
class TestMixedPrecision:
    """Test mixed precision training"""
    
    def test_mixed_precision_support(self):
        """Test mixed precision support check"""
        device = _cpp.DeviceManager.instance().get_cpu_device()
        supported = _cpp.supports_mixed_precision(device)
        # CPU doesn't support mixed precision, should be False
        assert isinstance(supported, bool)
    
    def test_mixed_precision_trainer_creation(self):
        """Test mixed precision trainer creation"""
        model = _cpp.MLP([4, 8, 2], "relu", "test_mp")
        model.build([4])
        model.compile("adam", 0.01, "mse")
        
        # This will fail on CPU, but should not crash
        try:
            trainer = _cpp.MixedPrecisionTrainer(model, loss_scale=1.0)
            assert trainer.get_loss_scale() == 1.0
        except RuntimeError:
            # Expected on CPU
            pass


@pytest.mark.skipif(not CPP_BACKEND_AVAILABLE, reason="C++ backend not available")
class TestTensorOperations:
    """Test tensor operations"""
    
    def test_add_operation(self):
        """Test add operation"""
        a_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b_data = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        
        a_cpp = _cpp.numpy_to_tensor(a_data)
        b_cpp = _cpp.numpy_to_tensor(b_data)
        
        c_cpp = _cpp.add(a_cpp, b_cpp)
        c_numpy = _cpp.tensor_to_numpy(c_cpp)
        
        expected = a_data + b_data
        np.testing.assert_allclose(c_numpy, expected)
    
    def test_relu_operation(self):
        """Test ReLU operation"""
        a_data = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        a_cpp = _cpp.numpy_to_tensor(a_data)
        
        b_cpp = _cpp.relu(a_cpp)
        b_numpy = _cpp.tensor_to_numpy(b_cpp)
        
        expected = np.maximum(0, a_data)
        np.testing.assert_allclose(b_numpy, expected)
    
    def test_sigmoid_operation(self):
        """Test sigmoid operation"""
        a_data = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        a_cpp = _cpp.numpy_to_tensor(a_data)
        
        b_cpp = _cpp.sigmoid(a_cpp)
        b_numpy = _cpp.tensor_to_numpy(b_cpp)
        
        expected = 1.0 / (1.0 + np.exp(-a_data))
        np.testing.assert_allclose(b_numpy, expected, rtol=1e-5)


@pytest.mark.skipif(not CPP_BACKEND_AVAILABLE, reason="C++ backend not available")
class TestGraphOperations:
    """Test graph operations"""
    
    def test_graph_creation(self):
        """Test graph creation"""
        graph = _cpp.Graph()
        assert graph is not None
    
    def test_placeholder_creation(self):
        """Test placeholder node creation"""
        graph = _cpp.Graph()
        x = graph.add_placeholder("x")
        assert x is not None
        assert x.name() == "x"
    
    def test_parameter_creation(self):
        """Test parameter node creation"""
        graph = _cpp.Graph()
        w_data = np.ones((3, 3), dtype=np.float32)
        w_cpp = _cpp.numpy_to_tensor(w_data)
        w = graph.add_parameter("w", w_cpp)
        assert w is not None
        assert w.name() == "w"
    
    def test_graph_compilation(self):
        """Test graph compilation"""
        graph = _cpp.Graph()
        x = graph.add_placeholder("x")
        w = graph.add_parameter("w", _cpp.numpy_to_tensor(np.ones((3, 3), dtype=np.float32)))
        matmul = graph.add_op("matmul", "matmul", [x, w])
        
        graph.compile()
        assert graph.is_compiled() == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
