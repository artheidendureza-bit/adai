"""
Optimizer implementations for adAI library
"""

import numpy as np
from typing import Optional
from .backend import to_numpy, from_numpy


class Optimizer:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.t = 0
    
    def update(self, params, grads):
        self.t += 1
        return self._update(params, grads)
    
    def _update(self, params, grads):
        """Subclasses implement this"""
        raise NotImplementedError

class SGD(Optimizer):    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
    
    def _update(self, params, grads):
        params_np = to_numpy(params)
        grads_np = to_numpy(grads)
        
        if self.momentum > 0:
            if self.velocity is None:
                self.velocity = np.zeros_like(params_np)
            self.velocity = self.momentum * self.velocity - self.learning_rate * grads_np
            updated_params = params_np + self.velocity
        else:
            updated_params = params_np - self.learning_rate * grads_np
        
        return from_numpy(updated_params)

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

    def _update(self, params, grads):
        params_np = to_numpy(params)
        grads_np = to_numpy(grads)
        
        if self.m is None:
            self.m = np.zeros_like(params_np)
            self.v = np.zeros_like(params_np)
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads_np
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads_np ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        updated_params = params_np - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return from_numpy(updated_params)

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
    
    def _update(self, params, grads):
        params_np = to_numpy(params)
        params_np = params_np - self.learning_rate * self.weight_decay * params_np
        params = from_numpy(params_np)
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
    
    def _update(self, params, grads):
        params_np = to_numpy(params)
        grads_np = to_numpy(grads)
        
        if self.m is None:
            self.m = np.zeros_like(params_np)
            self.v = np.zeros_like(params_np)
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads_np
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads_np ** 2)
        
        rho_inf = self.inf
        rho_t = rho_inf - 2 * self.t * (self.beta2 ** self.t) / (1 - self.beta2 ** self.t)
        
        if rho_t > 4:
            r_t = np.sqrt(
                (rho_t - 4) * (rho_t - 2) * self.inf /
                ((self.inf - 4) * (self.inf - 2) * rho_t)
            )
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            updated_params = params_np - self.learning_rate * r_t * m_hat / (np.sqrt(v_hat) + self.epsilon)
        else:
            updated_params = params_np - self.learning_rate * self.m
        
        return from_numpy(updated_params)

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
    
    def _update(self, params, grads):
        params_np = to_numpy(params)
        params_np = params_np - self.learning_rate * self.weight_decay * params_np
        params = from_numpy(params_np)
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