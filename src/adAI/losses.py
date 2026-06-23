"""
Loss functions for adAI library
"""

import numpy as np
from typing import Callable
from .backend import sub, mul, pow, mean, log, sum as backend_sum, to_numpy, from_numpy, div, add


def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error loss
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MSE loss
    """
    diff = sub(y_true, y_pred)
    squared = pow(diff, 2)
    return mean(squared)


def mean_squared_error_derivative(y_true, y_pred):
    """
    Derivative of MSE loss
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Gradient of MSE loss
    """
    diff = sub(y_pred, y_true)
    scaled = mul(diff, 2)
    y_true_np = to_numpy(y_true)
    size = y_true_np.size
    return div(scaled, size)


def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error loss
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE loss
    """
    diff = sub(y_true, y_pred)
    diff_np = to_numpy(diff)
    abs_diff = np.abs(diff_np)
    return from_numpy(np.mean(abs_diff))


def mean_absolute_error_derivative(y_true, y_pred):
    """
    Derivative of MAE loss
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Gradient of MAE loss
    """
    diff = sub(y_pred, y_true)
    diff_np = to_numpy(diff)
    sign = np.sign(diff_np)
    y_true_np = to_numpy(y_true)
    size = y_true_np.size
    return from_numpy(sign / size)


def binary_cross_entropy(y_true, y_pred, epsilon: float = 1e-15):
    """
    Binary Cross-Entropy loss
    
    Args:
        y_true: True values (0 or 1)
        y_pred: Predicted probabilities
        epsilon: Small value to avoid log(0)
        
    Returns:
        Binary cross-entropy loss
    """
    y_pred_np = to_numpy(y_pred)
    y_pred_clipped = np.clip(y_pred_np, epsilon, 1 - epsilon)
    y_true_np = to_numpy(y_true)
    
    term1 = mul(y_true, log(from_numpy(y_pred_clipped)))
    one_minus_y_pred = sub(1.0, from_numpy(y_pred_clipped))
    one_minus_y_true = sub(1.0, y_true)
    term2 = mul(one_minus_y_true, log(one_minus_y_pred))
    
    neg_sum = sub(0.0, add(term1, term2))
    return mean(neg_sum)


def binary_cross_entropy_derivative(y_true, y_pred, epsilon: float = 1e-15):
    """
    Derivative of Binary Cross-Entropy loss
    
    Args:
        y_true: True values (0 or 1)
        y_pred: Predicted probabilities
        epsilon: Small value to avoid division by zero
        
    Returns:
        Gradient of binary cross-entropy loss
    """
    y_pred_np = to_numpy(y_pred)
    y_pred_clipped = np.clip(y_pred_np, epsilon, 1 - epsilon)
    y_true_np = to_numpy(y_true)
    
    numerator = sub(from_numpy(y_pred_clipped), y_true)
    y_pred_times_one_minus = mul(from_numpy(y_pred_clipped), sub(1.0, from_numpy(y_pred_clipped)))
    size = y_true_np.size
    denominator = mul(y_pred_times_one_minus, size)
    
    return div(numerator, denominator)


def categorical_cross_entropy(y_true, y_pred, epsilon: float = 1e-15):
    """
    Categorical Cross-Entropy loss for multi-class classification
    
    Args:
        y_true: True values (one-hot encoded)
        y_pred: Predicted probabilities (softmax output)
        epsilon: Small value to avoid log(0)
        
    Returns:
        Categorical cross-entropy loss
    """
    y_pred_np = to_numpy(y_pred)
    y_pred_clipped = np.clip(y_pred_np, epsilon, 1 - epsilon)
    y_true_np = to_numpy(y_true)
    
    log_pred = log(from_numpy(y_pred_clipped))
    product = mul(y_true, log_pred)
    sum_axis = backend_sum(product, [-1])
    neg_sum = sub(0.0, mean(sum_axis))
    
    return neg_sum


def categorical_cross_entropy_derivative(y_true, y_pred, epsilon: float = 1e-15):
    """
    Derivative of Categorical Cross-Entropy loss
    
    Args:
        y_true: True values (one-hot encoded)
        y_pred: Predicted probabilities (softmax output)
        epsilon: Small value to avoid division by zero
        
    Returns:
        Gradient of categorical cross-entropy loss
    """
    y_pred_np = to_numpy(y_pred)
    y_pred_clipped = np.clip(y_pred_np, epsilon, 1 - epsilon)
    y_true_np = to_numpy(y_true)
    
    diff = sub(from_numpy(y_pred_clipped), y_true)
    size = y_true_np.size
    return div(diff, size)


def hinge_loss(y_true, y_pred):
    """
    Hinge loss for SVM-style classification
    
    Args:
        y_true: True values (-1 or 1)
        y_pred: Predicted values
        
    Returns:
        Hinge loss
    """
    y_true_np = to_numpy(y_true)
    y_pred_np = to_numpy(y_pred)
    product = mul(y_true, y_pred)
    one_minus = sub(1.0, product)
    one_minus_np = to_numpy(one_minus)
    max_val = np.maximum(0, one_minus_np)
    return from_numpy(np.mean(max_val))


def hinge_loss_derivative(y_true, y_pred):
    """
    Derivative of Hinge loss
    
    Args:
        y_true: True values (-1 or 1)
        y_pred: Predicted values
        
    Returns:
        Gradient of hinge loss
    """
    y_true_np = to_numpy(y_true)
    y_pred_np = to_numpy(y_pred)
    mask = (y_true_np * y_pred_np < 1).astype(float)
    size = y_true_np.size
    return from_numpy(-mask * y_true_np / size)


# Loss function registry
LOSS_FUNCTIONS: dict[str, tuple[Callable, Callable]] = {
    "mse": (mean_squared_error, mean_squared_error_derivative),
    "mae": (mean_absolute_error, mean_absolute_error_derivative),
    "binary_crossentropy": (binary_cross_entropy, binary_cross_entropy_derivative),
    "categorical_crossentropy": (categorical_cross_entropy, categorical_cross_entropy_derivative),
    "hinge": (hinge_loss, hinge_loss_derivative),
}


def get_loss(name: str) -> tuple[Callable, Callable]:
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
