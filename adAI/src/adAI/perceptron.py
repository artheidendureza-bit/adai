"""
Perceptron module for adAI
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from .base import BaseModel
from .activations import get_activation


class Perceptron(BaseModel):
    """A simple perceptron neural network unit"""
    
    def __init__(
        self,
        input_size: int,
        learning_rate: float = 0.01,
        epochs: int = 100,
        activation: str = "sigmoid"
    ):
        """
        Initialize a Perceptron
        
        Args:
            input_size: Number of input features
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            activation: Activation function ("sigmoid", "relu", "tanh")
            
        Raises:
            ValueError: If input_size <= 0 or activation is invalid
        """
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
            
        super().__init__(learning_rate=learning_rate, epochs=epochs)
        self.input_size = input_size
        self.activation = activation
        
        # Get activation functions
        self._activation_fn, self._activation_derivative_fn = get_activation(activation)
        
        # Initialize weights and bias
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0
    
    def summary(self) -> None:
        """Print model summary"""
        print(f"Perceptron Summary:")
        print(f"  Input size:    {self.input_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Epochs:        {self.epochs}")
        print(f"  Activation:    {self.activation}")
        super().summary()
    
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        return self._activation_fn(x)
    
    def _activate_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of activation function"""
        return self._activation_derivative_fn(x)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the perceptron"""
        z = np.dot(X, self.weights) + self.bias
        return self._activate(z)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
        early_stopping: bool = False,
        patience: int = 10,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Train the perceptron with optional validation and early stopping
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target values (n_samples,)
            X_val: Validation data (optional)
            y_val: Validation targets (optional)
            batch_size: Batch size for mini-batch training (optional, defaults to full batch)
            early_stopping: Whether to use early stopping (requires validation data)
            patience: Number of epochs with no improvement to stop training
            verbose: Print training progress
        
        Returns:
            Dictionary with training history and metrics
            
        Raises:
            ValueError: If data is invalid
        """
        self._validate_training_data(X, y, X_val, y_val)
        
        if batch_size is None:
            batch_size = len(X)
        
        val_loss_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X))
            epoch_loss = 0
            
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Calculate error
                error = y_batch - output
                batch_loss = np.mean(error**2)
                epoch_loss += batch_loss
                
                # Backward pass
                d_error = error * self._activate_derivative(output)
                
                # Update weights and bias
                self.weights += self.learning_rate * np.dot(X_batch.T, d_error) / len(X_batch)
                self.bias += self.learning_rate * np.mean(d_error)
            
            # Average loss for epoch
            epoch_loss /= max(1, len(X) // batch_size)
            self.loss_history.append(epoch_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_output = self.forward(X_val)
                val_error = y_val - val_output
                val_loss = np.mean(val_error**2)
                val_loss_history.append(val_loss)
                
                # Early stopping check
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch + 1}")
                            break
            
            if verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                msg = f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.6f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.6f}"
                print(msg)
        
        return self._get_training_history(val_loss_history)
    
    def print_training_summary(self, history: Dict[str, Any]) -> None:
        """Print training results summary"""
        print(f"\nTraining completed in {history['epochs_trained']} epochs")
        print(f"  Final train loss: {history['final_loss']:.6f}")
        if history['final_val_loss']:
            print(f"  Final val loss:   {history['final_val_loss']:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        return self.forward(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the perceptron on test data
        
        Args:
            X: Test data (n_samples, n_features)
            y: Test targets (n_samples,)
        
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X)
        return self._compute_metrics(y, predictions)
    
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print evaluation metrics"""
        print(f"\nTest Metrics:")
        print(f"  MSE:      {metrics['mse']:.6f}")
        print(f"  MAE:      {metrics['mae']:.6f}")
        print(f"  RMSE:     {metrics['rmse']:.6f}")
        if metrics['accuracy']:
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
    
    def save(self, path: str) -> None:
        """Save perceptron parameters to a compressed file."""
        np.savez_compressed(
            path,
            input_size=self.input_size,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            activation=self.activation,
            weights=self.weights,
            bias=self.bias,
            loss_history=np.array(self.loss_history, dtype=np.float64),
        )
    
    @classmethod
    def load(cls, path: str) -> "Perceptron":
        """Load a saved perceptron from a compressed file."""
        data = np.load(path, allow_pickle=True)
        perceptron = cls(
            input_size=int(data["input_size"]),
            learning_rate=float(data["learning_rate"]),
            epochs=int(data["epochs"]),
            activation=str(data["activation"]),
        )
        perceptron.weights = data["weights"]
        perceptron.bias = float(data["bias"])
        perceptron.loss_history = data["loss_history"].tolist()
        return perceptron
    
    def generate_data(
        self,
        n_train: int = 200,
        n_val: int = 50,
        n_test: int = 50,
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic training, validation, and test data
        
        Args:
            n_train: Number of training samples (default: 200)
            n_val: Number of validation samples (default: 50)
            n_test: Number of test samples (default: 50)
            seed: Random seed for reproducibility (default: 42)
        
        Returns:
            Dictionary with 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
        
        Example:
            >>> data = perceptron.generate_data(n_train=300)
            >>> history = perceptron.train(data['X_train'], data['y_train'], ...)
        """
        np.random.seed(seed)
        
        # Generate training data
        X_train = np.random.randn(n_train, self.input_size).astype(np.float32)
        y_train = (X_train.sum(axis=1) > 0).astype(float)
        
        # Generate validation data
        X_val = np.random.randn(n_val, self.input_size).astype(np.float32)
        y_val = (X_val.sum(axis=1) > 0).astype(float)
        
        # Generate test data
        X_test = np.random.randn(n_test, self.input_size).astype(np.float32)
        y_test = (X_test.sum(axis=1) > 0).astype(float)
        
        print(f"\nData generated:")
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }


def generate_logic_gate_data(gate: str):
    """Return input/output data for a supported logic gate."""
    gate = gate.strip().lower()
    if gate == "not":
        X = np.array([[0], [1]], dtype=np.float32)
        y = np.array([1, 0], dtype=float)
    elif gate == "or":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([0, 1, 1, 1], dtype=float)
    elif gate == "nor":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([1, 0, 0, 0], dtype=float)
    elif gate == "and":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([0, 0, 0, 1], dtype=float)
    elif gate == "nand":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([1, 1, 1, 0], dtype=float)
    else:
        raise ValueError("Unsupported logic gate. Use NOT, OR, NOR, AND, or NAND.")
    return X, y


def generate_not_data():
    return generate_logic_gate_data("not")


def generate_or_data():
    return generate_logic_gate_data("or")


def generate_nor_data():
    return generate_logic_gate_data("nor")


def generate_and_data():
    return generate_logic_gate_data("and")


def generate_nand_data():
    return generate_logic_gate_data("nand")


def GeneratePerceptron(
    input_size: int,
    learning_rate: float = 0.01,
    epochs: int = 100,
    activation: str = "sigmoid"
) -> Perceptron:
    """
    Automatically generate and initialize a perceptron
    
    Args:
        input_size: Number of input features
        learning_rate: Learning rate for training (default: 0.01)
        epochs: Number of training epochs (default: 100)
        activation: Activation function - "sigmoid", "relu", or "tanh" (default: "sigmoid")
    
    Returns:
        A ready-to-use Perceptron instance
    
    Example:
        >>> perceptron = GeneratePerceptron(input_size=5)
        >>> perceptron.train(X_train, y_train)
        >>> predictions = perceptron.predict(X_test)
    """
    return Perceptron(
        input_size=input_size,
        learning_rate=learning_rate,
        epochs=epochs,
        activation=activation
    )
