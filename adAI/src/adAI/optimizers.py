"""
Optimizer implementations for adAI library
"""

import numpy as np
from typing import Optional


class Optimizer:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.t = 0
    
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        self.t += 1
        return self._update(params, grads)
    
    def _update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Subclasses implement this"""
        raise NotImplementedError

class SGD(Optimizer):    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
    
    def _update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.momentum > 0:
            if self.velocity is None:
                self.velocity = np.zeros_like(params)
            self.velocity = self.momentum * self.velocity - self.learning_rate * grads
            return params + self.velocity
        else:
            return params - self.learning_rate * grads

class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def _update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class AdamW(Adam):    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(learning_rate, beta1, beta2, epsilon)
        self.weight_decay = weight_decay
    
    def _update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        params = params - self.learning_rate * self.weight_decay * params
        return super()._update(params, grads)


class RAdam(Adam):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate, beta1, beta2, epsilon)
        self.inf = 4.0 / (1.0 - beta2) - 2.0
    
    def _update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        rho_inf = self.inf
        rho_t = rho_inf - 2 * self.t * (self.beta2 ** self.t) / (1 - self.beta2 ** self.t)
        
        if rho_t > 4:
            r_t = np.sqrt(
                (rho_t - 4) * (rho_t - 2) * self.inf /
                ((self.inf - 4) * (self.inf - 2) * rho_t)
            )
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            return params - self.learning_rate * r_t * m_hat / (np.sqrt(v_hat) + self.epsilon)
        else:
            return params - self.learning_rate * self.m

class RAdamW(RAdam):    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(learning_rate, beta1, beta2, epsilon)
        self.weight_decay = weight_decay
    
    def _update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        params = params - self.learning_rate * self.weight_decay * params
        return super()._update(params, grads)


def get_optimizer(name: str, learning_rate: float = 0.01, **kwargs) -> Optimizer:
    name = name.lower()
    optimizers = {
        "sgd": SGD,
        "adam": Adam,
        "adamw": AdamW,
        "radam": RAdam,
        "radamw": RAdamW,
    }
    
    if name not in optimizers:
        raise ValueError(
            f"Unknown optimizer: {name}. "
            f"Available: {list(optimizers.keys())}"
        )
    
    return optimizers[name](learning_rate=learning_rate, **kwargs)