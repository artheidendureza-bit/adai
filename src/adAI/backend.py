"""
Backend management for adAI.
Provides interface to switch between Python (NumPy) and C++ backends.
"""

import os
import sys
from typing import Optional, Any
import numpy as np

# Try to import C++ backend
CPP_BACKEND_AVAILABLE = False
_cpp_module = None

try:
    # Try to import the compiled C++ module
    import adaicpp
    _cpp_module = adaicpp
    CPP_BACKEND_AVAILABLE = True
except ImportError:
    CPP_BACKEND_AVAILABLE = False


class Backend:
    """Backend types."""
    NUMPY = "numpy"
    CPP = "cpp"


# Current backend setting
_current_backend = Backend.CPP if CPP_BACKEND_AVAILABLE else Backend.NUMPY


def set_backend(backend: str) -> None:
    """
    Set the computation backend.
    
    Args:
        backend: Either 'numpy' or 'cpp'
    
    Raises:
        ValueError: If backend is invalid or C++ backend is requested but not available
    """
    global _current_backend
    
    if backend == Backend.CPP and not CPP_BACKEND_AVAILABLE:
        raise ValueError(
            "C++ backend is not available. "
            "Please build the C++ backend with Python bindings enabled."
        )
    
    if backend not in [Backend.NUMPY, Backend.CPP]:
        raise ValueError(f"Invalid backend: {backend}. Must be 'numpy' or 'cpp'")
    
    _current_backend = backend


def get_backend() -> str:
    """Get the current backend."""
    return _current_backend


def is_cpp_available() -> bool:
    """Check if C++ backend is available."""
    return CPP_BACKEND_AVAILABLE


def get_cpp_module() -> Optional[Any]:
    """Get the C++ module if available."""
    return _cpp_module


def initialize_cpp() -> None:
    """Initialize the C++ backend."""
    if CPP_BACKEND_AVAILABLE and _cpp_module is not None:
        _cpp_module.initialize()


# Auto-initialize C++ backend if available
if CPP_BACKEND_AVAILABLE:
    try:
        initialize_cpp()
    except Exception as e:
        print(f"Warning: Failed to initialize C++ backend: {e}")
        CPP_BACKEND_AVAILABLE = False
        _current_backend = Backend.NUMPY


# Tensor creation functions that work with both backends
def zeros(shape, device=None):
    """Create a tensor filled with zeros."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        dev = _cpp_module.cpu_device() if device is None else device
        return _cpp_module.Tensor.zeros(shape, dev)
    else:
        return np.zeros(shape, dtype=np.float32)


def ones(shape, device=None):
    """Create a tensor filled with ones."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        dev = _cpp_module.cpu_device() if device is None else device
        return _cpp_module.Tensor.ones(shape, dev)
    else:
        return np.ones(shape, dtype=np.float32)


def randn(shape, device=None):
    """Create a tensor with random normal values."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        dev = _cpp_module.cpu_device() if device is None else device
        return _cpp_module.Tensor.randn(shape, dev)
    else:
        return np.random.randn(*shape).astype(np.float32)


# Tensor operations that work with both backends
def add(a, b):
    """Element-wise addition."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        return _cpp_module.add(a, b)
    else:
        return np.add(a, b)


def sub(a, b):
    """Element-wise subtraction."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        return _cpp_module.sub(a, b)
    else:
        return np.subtract(a, b)


def mul(a, b):
    """Element-wise multiplication."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        return _cpp_module.mul(a, b)
    else:
        return np.multiply(a, b)


def div(a, b):
    """Element-wise division."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        return _cpp_module.div(a, b)
    else:
        return np.divide(a, b)


def matmul(a, b):
    """Matrix multiplication."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        return _cpp_module.matmul(a, b)
    else:
        return np.matmul(a, b)


def relu(x):
    """ReLU activation."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        return _cpp_module.relu(x)
    else:
        return np.maximum(0, x)


def sigmoid(x):
    """Sigmoid activation."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        return _cpp_module.sigmoid(x)
    else:
        return 1 / (1 + np.exp(-x))


def tanh(x):
    """Tanh activation."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        return _cpp_module.tanh(x)
    else:
        return np.tanh(x)


def softmax(x, axis=-1):
    """Softmax activation."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        return _cpp_module.softmax(x)
    else:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def exp(x):
    """Element-wise exponential."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        return _cpp_module.exp(x)
    else:
        return np.exp(x)


def log(x):
    """Element-wise natural logarithm."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        return _cpp_module.log(x)
    else:
        return np.log(x)


def pow(x, exponent):
    """Element-wise power."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        return _cpp_module.pow(x, exponent)
    else:
        return np.power(x, exponent)


def sum(x, axis=None, keepdims=False):
    """Sum reduction."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        axes = [] if axis is None else [axis] if isinstance(axis, int) else list(axis)
        return _cpp_module.sum(x, axes, keepdims)
    else:
        return np.sum(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    """Mean reduction."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        axes = [] if axis is None else [axis] if isinstance(axis, int) else list(axis)
        return _cpp_module.mean(x, axes, keepdims)
    else:
        return np.mean(x, axis=axis, keepdims=keepdims)


# Conversion utilities
def to_numpy(x):
    """Convert tensor to numpy array."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        if hasattr(x, 'numpy'):
            return x.numpy()
        else:
            return np.array(x)
    else:
        return np.asarray(x)


def from_numpy(x, device=None):
    """Convert numpy array to tensor."""
    if _current_backend == Backend.CPP and CPP_BACKEND_AVAILABLE:
        dev = _cpp_module.cpu_device() if device is None else device
        return _cpp_module.from_numpy(np.asarray(x, dtype=np.float32), dev)
    else:
        return np.asarray(x, dtype=np.float32)
