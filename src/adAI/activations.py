"""
Shared activation functions for adAI library
"""
import numpy as np
from typing import Callable
from .backend import sigmoid, relu, tanh as tanh_backend, softmax as softmax_backend, to_numpy, ones, mul, sub

SIGMOID_CLIP_MIN = -500
SIGMOID_CLIP_MAX = 500

def sigmoid_activation(x):
    """Sigmoid activation using backend."""
    return sigmoid(x)

def sigmoid_derivative(x):
    """Sigmoid derivative (assumes x is already sigmoid output)."""
    return mul(x, sub(1.0, x))

def relu_activation(x):
    """ReLU activation using backend."""
    return relu(x)

def relu_derivative(x):
    """ReLU derivative (assumes x is input to ReLU)."""
    x_np = to_numpy(x)
    return (x_np > 0).astype(float)

def tanh_activation(x):
    """Tanh activation using backend."""
    return tanh_backend(x)

def tanh_derivative(x):
    """Tanh derivative (assumes x is already tanh output)."""
    x_np = to_numpy(x)
    return sub(1.0, mul(x_np, x_np))

def linear(x):
    """Linear activation (identity)."""
    return x

def linear_derivative(x):
    """Linear derivative (all ones)."""
    return ones(x.shape)

def softmax_activation(x):
    """Softmax activation using backend."""
    return softmax_backend(x)

# Export softmax for direct import
softmax = softmax_activation

ACTIVATION_FUNCTIONS: dict[str, tuple[Callable, Callable]] = {
    "sigmoid": (sigmoid_activation, sigmoid_derivative),
    "relu": (relu_activation, relu_derivative),
    "tanh": (tanh_activation, tanh_derivative),
    "linear": (linear, linear_derivative),
    "softmax": (softmax_activation, None),  # Softmax derivative is handled separately
}

def get_activation(name: str) -> tuple[Callable, Callable]:
    name = name.lower()
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(
            f"Unknown activation function: {name}. "
            f"Available: {list(ACTIVATION_FUNCTIONS.keys())}"
        )
    return ACTIVATION_FUNCTIONS[name]