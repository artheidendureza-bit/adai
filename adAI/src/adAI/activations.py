"""
Shared activation functions for adAI library
"""
import numpy as np
from typing import Callable

SIGMOID_CLIP_MIN = -500
SIGMOID_CLIP_MAX = 500

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, SIGMOID_CLIP_MIN, SIGMOID_CLIP_MAX)))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def tanh_activation(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - x**2

def linear(x: np.ndarray) -> np.ndarray:
    return x

def linear_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

ACTIVATION_FUNCTIONS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "tanh": (tanh_activation, tanh_derivative),
    "linear": (linear, linear_derivative),
}

def get_activation(name: str) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    name = name.lower()
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(
            f"Unknown activation function: {name}. "
            f"Available: {list(ACTIVATION_FUNCTIONS.keys())}"
        )
    return ACTIVATION_FUNCTIONS[name]