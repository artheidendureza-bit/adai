"""
Additional layers for adAI library
"""

import numpy as np
from typing import Optional


class BatchNormLayer:    
    def __init__(self, num_features: int, epsilon: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        
        self.X_centered = None
        self.std = None
        self.X_normalized = None
        
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if X.ndim == 4:
            axes = (0, 1, 2)
            shape = (1, 1, 1, self.num_features)
        else:
            axes = 0
            shape = (1, self.num_features)
        
        if training:
            batch_mean = np.mean(X, axis=axes, keepdims=True)
            batch_var = np.var(X, axis=axes, keepdims=True)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * np.squeeze(batch_mean)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * np.squeeze(batch_var)
            
            self.std = np.sqrt(batch_var + self.epsilon)
            self.X_centered = X - batch_mean
            self.X_normalized = self.X_centered / self.std
        else:
            batch_mean = self.running_mean.reshape(shape)
            batch_var = self.running_var.reshape(shape)
            self.std = np.sqrt(batch_var + self.epsilon)
            self.X_centered = X - batch_mean
            self.X_normalized = self.X_centered / self.std
        
        out = self.gamma.reshape(shape) * self.X_normalized + self.beta.reshape(shape)
        return out
    
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self.X_normalized is None:
            raise RuntimeError("Must call forward with training=True before backward")
        
        if d_out.ndim == 4:
            axes = (0, 1, 2)
            batch_size = d_out.shape[0] * d_out.shape[1] * d_out.shape[2]
        else:
            axes = 0
            batch_size = d_out.shape[0]
        
        d_gamma = np.sum(d_out * self.X_normalized, axis=axes)
        d_beta = np.sum(d_out, axis=axes)
        
        d_x_normalized = d_out * self.gamma
        
        d_var = np.sum(d_x_normalized * self.X_centered, axis=axes, keepdims=True) * -0.5 * (self.std ** -3)
        d_mean = np.sum(d_x_normalized, axis=axes, keepdims=True) * -1.0 / self.std + d_var * np.mean(-2 * self.X_centered, axis=axes, keepdims=True)
        d_x = d_x_normalized / self.std + d_var * 2 * self.X_centered / batch_size + d_mean / batch_size
        
        self.d_gamma = d_gamma
        self.d_beta = d_beta
        
        return d_x
    
    def update_parameters(self, learning_rate: float):
        """Update learnable parameters"""
        self.gamma -= learning_rate * self.d_gamma
        self.beta -= learning_rate * self.d_beta


class DropoutLayer:    
    def __init__(self, dropout_rate: float = 0.5):
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError("dropout_rate must be in [0, 1)")
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if training and self.dropout_rate > 0:
            self.mask = (np.random.rand(*X.shape) >= self.dropout_rate).astype(np.float32) / (1 - self.dropout_rate)
            return X * self.mask
        else:
            return X
    
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self.mask is not None:
            return d_out * self.mask
        else:
            return d_out
