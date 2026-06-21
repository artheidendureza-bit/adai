"""
Loss functions for adAI library
"""

import numpy as np
from typing import Callable


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Mean Squared Error loss
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MSE loss
    """
    return np.mean((y_true - y_pred) ** 2)


def mean_squared_error_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of MSE loss
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Gradient of MSE loss
    """
    return 2 * (y_pred - y_true) / y_true.size


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Mean Absolute Error loss
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE loss
    """
    return np.mean(np.abs(y_true - y_pred))


def mean_absolute_error_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of MAE loss
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Gradient of MAE loss
    """
    return np.sign(y_pred - y_true) / y_true.size


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    """
    Binary Cross-Entropy loss
    
    Args:
        y_true: True values (0 or 1)
        y_pred: Predicted probabilities
        epsilon: Small value to avoid log(0)
        
    Returns:
        Binary cross-entropy loss
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    """
    Derivative of Binary Cross-Entropy loss
    
    Args:
        y_true: True values (0 or 1)
        y_pred: Predicted probabilities
        epsilon: Small value to avoid division by zero
        
    Returns:
        Gradient of binary cross-entropy loss
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    """
    Categorical Cross-Entropy loss for multi-class classification
    
    Args:
        y_true: True values (one-hot encoded)
        y_pred: Predicted probabilities (softmax output)
        epsilon: Small value to avoid log(0)
        
    Returns:
        Categorical cross-entropy loss
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))


def categorical_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    """
    Derivative of Categorical Cross-Entropy loss
    
    Args:
        y_true: True values (one-hot encoded)
        y_pred: Predicted probabilities (softmax output)
        epsilon: Small value to avoid division by zero
        
    Returns:
        Gradient of categorical cross-entropy loss
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / y_true.size


def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Hinge loss for SVM-style classification
    
    Args:
        y_true: True values (-1 or 1)
        y_pred: Predicted values
        
    Returns:
        Hinge loss
    """
    return np.mean(np.maximum(0, 1 - y_true * y_pred))


def hinge_loss_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of Hinge loss
    
    Args:
        y_true: True values (-1 or 1)
        y_pred: Predicted values
        
    Returns:
        Gradient of hinge loss
    """
    mask = (y_true * y_pred < 1).astype(float)
    return -mask * y_true / y_true.size


# Loss function registry
LOSS_FUNCTIONS: dict[str, tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]]] = {
    "mse": (mean_squared_error, mean_squared_error_derivative),
    "mae": (mean_absolute_error, mean_absolute_error_derivative),
    "binary_crossentropy": (binary_cross_entropy, binary_cross_entropy_derivative),
    "categorical_crossentropy": (categorical_cross_entropy, categorical_cross_entropy_derivative),
    "hinge": (hinge_loss, hinge_loss_derivative),
}


def get_loss(name: str) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """
    Get loss function and its derivative by name
    
    Args:
        name: Name of loss function
        
    Returns:
        Tuple of (loss_function, derivative_function)
        
    Raises:
        ValueError: If loss name is not recognized
    """
    name = name.lower()
    if name not in LOSS_FUNCTIONS:
        raise ValueError(
            f"Unknown loss function: {name}. "
            f"Available: {list(LOSS_FUNCTIONS.keys())}"
        )
    return LOSS_FUNCTIONS[name]
