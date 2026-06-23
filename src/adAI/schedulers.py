"""
Learning rate schedulers for adAI library
"""

from abc import ABC, abstractmethod
import numpy as np


class LRScheduler(ABC):
    """Base class for learning rate schedulers"""
    
    @abstractmethod
    def get_lr(self, initial_lr: float, epoch: int) -> float:
        """
        Get learning rate for current epoch
        
        Args:
            initial_lr: Initial learning rate
            epoch: Current epoch number
            
        Returns:
            Learning rate for current epoch
        """
        pass


class ConstantLR(LRScheduler):
    """Constant learning rate (no scheduling)"""
    
    def get_lr(self, initial_lr: float, epoch: int) -> float:
        return initial_lr


class StepLR(LRScheduler):
    """Step learning rate decay
    
    Decays learning rate by gamma every step_size epochs
    """
    
    def __init__(self, step_size: int, gamma: float = 0.1):
        """
        Args:
            step_size: Number of epochs between decay steps
            gamma: Multiplicative factor of learning rate decay
        """
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self, initial_lr: float, epoch: int) -> float:
        return initial_lr * (self.gamma ** (epoch // self.step_size))


class ExponentialLR(LRScheduler):
    """Exponential learning rate decay
    
    Decays learning rate by gamma every epoch
    """
    
    def __init__(self, gamma: float = 0.95):
        """
        Args:
            gamma: Multiplicative factor of learning rate decay per epoch
        """
        self.gamma = gamma
    
    def get_lr(self, initial_lr: float, epoch: int) -> float:
        return initial_lr * (self.gamma ** epoch)


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate
    
    Sets learning rate using cosine annealing schedule
    """
    
    def __init__(self, T_max: int, eta_min: float = 0.0):
        """
        Args:
            T_max: Maximum number of iterations
            eta_min: Minimum learning rate
        """
        self.T_max = T_max
        self.eta_min = eta_min
    
    def get_lr(self, initial_lr: float, epoch: int) -> float:
        return self.eta_min + (initial_lr - self.eta_min) * (
            1 + np.cos(np.pi * epoch / self.T_max)
        ) / 2


class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving
    
    Monitors a quantity and if no improvement is seen for a 'patience'
    number of epochs, the learning rate is reduced
    """
    
    def __init__(
        self,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-6,
        threshold: float = 1e-4,
    ):
        """
        Args:
            mode: One of 'min' or 'max'. In 'min' mode, lr will be reduced when
                  the quantity monitored has stopped decreasing. In 'max' mode,
                  lr will be reduced when the quantity monitored has stopped increasing.
            factor: Factor by which the learning rate will be reduced
            patience: Number of epochs with no improvement after which learning rate will be reduced
            min_lr: Lower bound on the learning rate
            threshold: Threshold for measuring the new optimum
        """
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        
        self.best = None
        self.num_bad_epochs = 0
    
    def step(self, metrics: float, current_lr: float) -> float:
        """
        Update learning rate based on metrics
        
        Args:
            metrics: Current metric value
            current_lr: Current learning rate
            
        Returns:
            New learning rate
        """
        if self.best is None:
            self.best = metrics
            return current_lr
        
        if self.mode == 'min':
            improved = metrics < self.best - self.threshold
        else:
            improved = metrics > self.best + self.threshold
        
        if improved:
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            new_lr = max(current_lr * self.factor, self.min_lr)
            self.num_bad_epochs = 0
            return new_lr
        
        return current_lr
