"""
Base model class for adAI library
"""

import numpy as np
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import logging


# Set up logging
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Base class for all models in adAI library
    Provides common training, validation, and evaluation logic
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 100,
    ):
        """
        Initialize base model parameters
        
        Args:
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            
        Raises:
            ValueError: If learning_rate <= 0 or epochs <= 0
        """
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
            
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history: list[float] = []
    
    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the model
        
        Args:
            X: Input data
            
        Returns:
            Model output
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test data
        
        Args:
            X: Test data
            y: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model parameters to file
        
        Args:
            path: Path to save file
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """
        Load model from file
        
        Args:
            path: Path to saved model file
            
        Returns:
            Loaded model instance
        """
        pass
    
    def summary(self) -> None:
        """
        Print model summary
        """
        print(f"Model Summary:")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Epochs:        {self.epochs}")
    
    def _validate_training_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """
        Validate training data
        
        Args:
            X: Training data
            y: Training targets
            X_val: Optional validation data
            y_val: Optional validation targets
            
        Raises:
            ValueError: If data is invalid
        """
        if len(X) == 0:
            raise ValueError("Training data X cannot be empty")
        if len(y) == 0:
            raise ValueError("Training targets y cannot be empty")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length, got {len(X)} and {len(y)}")
        
        if X_val is not None and y_val is not None:
            if len(X_val) == 0:
                raise ValueError("Validation data X_val cannot be empty")
            if len(y_val) == 0:
                raise ValueError("Validation targets y_val cannot be empty")
            if len(X_val) != len(y_val):
                raise ValueError(f"X_val and y_val must have same length, got {len(X_val)} and {len(y_val)}")
        elif (X_val is None) != (y_val is None):
            raise ValueError("Both X_val and y_val must be provided together or both None")
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute common evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        y_true_flat = np.squeeze(y_true)
        y_pred_flat = np.squeeze(y_pred)
        
        mse = np.mean((y_true_flat - y_pred_flat) ** 2)
        mae = np.mean(np.abs(y_true_flat - y_pred_flat))
        
        # Check if binary classification
        is_binary = np.all((y_true_flat == 0) | (y_true_flat == 1))
        accuracy = None
        if is_binary:
            binary_predictions = (y_pred_flat > 0.5).astype(int)
            accuracy = np.mean(binary_predictions == y_true_flat)
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": np.sqrt(mse),
            "accuracy": accuracy,
        }
    
    def _get_training_history(
        self,
        val_loss_history: Optional[list[float]] = None,
    ) -> Dict[str, Any]:
        """
        Get training history dictionary
        
        Args:
            val_loss_history: Optional validation loss history
            
        Returns:
            Training history dictionary
        """
        return {
            "loss_history": self.loss_history,
            "val_loss_history": val_loss_history,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "final_val_loss": val_loss_history[-1] if val_loss_history else None,
            "epochs_trained": len(self.loss_history),
        }
