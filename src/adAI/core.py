"""
Core utilities for adAI library
"""

import numpy as np
from typing import Tuple, Optional


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets
    
    Args:
        X: Features array
        y: Target array
        test_size: Proportion of data for test set (default: 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    
    train_idx, test_idx = indices[:-n_test], indices[-n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def normalize(X: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Normalize data to [0, 1] range
    
    Args:
        X: Input array
        axis: Axis along which to normalize (None for global normalization)
        
    Returns:
        Normalized array
    """
    X_min = X.min(axis=axis, keepdims=True)
    X_max = X.max(axis=axis, keepdims=True)
    return (X - X_min) / (X_max - X_min + 1e-8)


def standardize(X: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Standardize data to zero mean and unit variance
    
    Args:
        X: Input array
        axis: Axis along which to standardize (None for global standardization)
        
    Returns:
        Standardized array
    """
    mean = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True)
    return (X - mean) / (std + 1e-8)
