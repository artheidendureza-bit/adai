"""
Additional layers for adAI library
"""

import numpy as np
from typing import Optional
from .backend import to_numpy, from_numpy, sub, mul, div, sum as backend_sum, mean, ones, zeros, randn


class BatchNormLayer:    
    def __init__(self, num_features: int, epsilon: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        self.gamma = ones([num_features])
        self.beta = zeros([num_features])
        
        self.running_mean = zeros([num_features])
        self.running_var = ones([num_features])
        
        self.X_centered = None
        self.std = None
        self.X_normalized = None
        
    def forward(self, X, training: bool = True):
        X_np = to_numpy(X)
        if X_np.ndim == 4:
            axes = (0, 1, 2)
            shape = (1, 1, 1, self.num_features)
        else:
            axes = 0
            shape = (1, self.num_features)
        
        if training:
            batch_mean = np.mean(X_np, axis=axes, keepdims=True)
            batch_var = np.var(X_np, axis=axes, keepdims=True)
            
            self.running_mean_np = to_numpy(self.running_mean)
            self.running_var_np = to_numpy(self.running_var)
            
            self.running_mean_np = (1 - self.momentum) * self.running_mean_np + self.momentum * np.squeeze(batch_mean)
            self.running_var_np = (1 - self.momentum) * self.running_var_np + self.momentum * np.squeeze(batch_var)
            
            self.running_mean = from_numpy(self.running_mean_np)
            self.running_var = from_numpy(self.running_var_np)
            
            std_np = np.sqrt(batch_var + self.epsilon)
            self.std = from_numpy(std_np)
            self.X_centered = sub(X, from_numpy(batch_mean))
            self.X_normalized = div(self.X_centered, self.std)
        else:
            batch_mean = self.running_mean_np.reshape(shape)
            batch_var = self.running_var_np.reshape(shape)
            std_np = np.sqrt(batch_var + self.epsilon)
            self.std = from_numpy(std_np)
            self.X_centered = sub(X, from_numpy(batch_mean))
            self.X_normalized = div(self.X_centered, self.std)
        
        gamma_np = to_numpy(self.gamma)
        beta_np = to_numpy(self.beta)
        X_normalized_np = to_numpy(self.X_normalized)
        
        out_np = gamma_np.reshape(shape) * X_normalized_np + beta_np.reshape(shape)
        return from_numpy(out_np)
    
    def backward(self, d_out):
        if self.X_normalized is None:
            raise RuntimeError("Must call forward with training=True before backward")
        
        d_out_np = to_numpy(d_out)
        X_normalized_np = to_numpy(self.X_normalized)
        X_centered_np = to_numpy(self.X_centered)
        std_np = to_numpy(self.std)
        gamma_np = to_numpy(self.gamma)
        
        if d_out_np.ndim == 4:
            axes = (0, 1, 2)
            batch_size = d_out_np.shape[0] * d_out_np.shape[1] * d_out_np.shape[2]
        else:
            axes = 0
            batch_size = d_out_np.shape[0]
        
        d_gamma = np.sum(d_out_np * X_normalized_np, axis=axes)
        d_beta = np.sum(d_out_np, axis=axes)
        
        d_x_normalized = d_out_np * gamma_np
        
        d_var = np.sum(d_x_normalized * X_centered_np, axis=axes, keepdims=True) * -0.5 * (std_np ** -3)
        d_mean = np.sum(d_x_normalized, axis=axes, keepdims=True) * -1.0 / std_np + d_var * np.mean(-2 * X_centered_np, axis=axes, keepdims=True)
        d_x_np = d_x_normalized / std_np + d_var * 2 * X_centered_np / batch_size + d_mean / batch_size
        
        self.d_gamma = d_gamma
        self.d_beta = d_beta
        
        return from_numpy(d_x_np)
    
    def update_parameters(self, learning_rate: float):
        """Update learnable parameters"""
        gamma_np = to_numpy(self.gamma)
        beta_np = to_numpy(self.beta)
        
        gamma_np -= learning_rate * self.d_gamma
        beta_np -= learning_rate * self.d_beta
        
        self.gamma = from_numpy(gamma_np)
        self.beta = from_numpy(beta_np)


class DropoutLayer:    
    def __init__(self, dropout_rate: float = 0.5):
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError("dropout_rate must be in [0, 1)")
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, X, training: bool = True):
        X_np = to_numpy(X)
        if training and self.dropout_rate > 0:
            self.mask = (np.random.rand(*X_np.shape) >= self.dropout_rate).astype(np.float32) / (1 - self.dropout_rate)
            return from_numpy(X_np * self.mask)
        else:
            return X
    
    def backward(self, d_out):
        d_out_np = to_numpy(d_out)
        if self.mask is not None:
            return from_numpy(d_out_np * self.mask)
        return d_out
