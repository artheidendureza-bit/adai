"""
MLP module for adAI
"""

import numpy as np
from typing import List, Optional, Dict, Any
from .base import BaseModel
from .activations import get_activation, softmax
from .perceptron import (
    generate_and_data,
    generate_or_data,
    generate_nor_data,
    generate_nand_data,
    generate_not_data,
)


class MLP(BaseModel):
    """A multi-layer perceptron network."""

    def __init__(
        self,
        input_size: int,
        hidden_layers: Optional[List[int]] = None,
        output_size: int = 1,
        learning_rate: float = 0.01,
        epochs: int = 100,
        activation: str = "relu",
        output_activation: str = "sigmoid",
        l2: float = 0.0,
        dropout: float = 0.0,
    ):
        """
        Initialize an MLP with dynamically built hidden layers.

        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            output_size: Number of output units
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            activation: Activation for hidden layers ("relu", "tanh", "sigmoid").
            output_activation: Activation for the output layer
                ("sigmoid", "relu", "linear", or "softmax"). Use "relu" or "linear"
                for regression tasks where targets can exceed [0, 1].
            l2: L2 regularization coefficient
            dropout: Dropout rate (must be in [0.0, 1.0))
            
        Raises:
            ValueError: If input_size <= 0, output_size <= 0, or dropout is invalid
        """
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0.0, 1.0)")
            
        super().__init__(learning_rate=learning_rate, epochs=epochs)
        self.input_size = input_size
        self.hidden_layers = hidden_layers or []
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        self.l2 = l2
        self.dropout = dropout

        # Get activation functions
        self._activation_fn, self._activation_derivative_fn = get_activation(activation)
        self._output_activation_fn, self._output_activation_derivative_fn = get_activation(output_activation)
        # Override with softmax if needed
        if output_activation == "softmax":
            self._output_activation_fn = softmax
            self._output_activation_derivative_fn = lambda x: x * (1 - x)  # Approximation

        self.layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        self.layers: List[Dict[str, np.ndarray]] = []
        for in_size, out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.layers.append({
                "weights": np.random.randn(in_size, out_size).astype(np.float32) * 0.01,
                "bias": np.zeros(out_size, dtype=np.float32),
            })

    def summary(self):
        """Print model summary."""
        total_params = sum(layer["weights"].size + layer["bias"].size for layer in self.layers)
        print("MLP Summary:")
        print(f"  Input size:          {self.input_size}")
        print(f"  Hidden layers:       {self.hidden_layers}")
        print(f"  Output size:         {self.output_size}")
        print(f"  Activation:          {self.activation}")
        print(f"  Output activation:   {self.output_activation}")
        print(f"  Layers:              {self.layer_sizes}")
        print(f"  Dropout:             {self.dropout}")
        print(f"  L2 regularizer:      {self.l2}")
        print(f"  Total params:        {total_params}")

    def _activation(self, x: np.ndarray) -> np.ndarray:
        return self._activation_fn(x)

    def _activation_derivative(self, activated: np.ndarray) -> np.ndarray:
        return self._activation_derivative_fn(activated)

    def _output_activation(self, x: np.ndarray) -> np.ndarray:
        return self._output_activation_fn(x)

    def _output_activation_derivative(self, activated: np.ndarray) -> np.ndarray:
        return self._output_activation_derivative_fn(activated)

    def forward(self, X: np.ndarray, training: bool = False):
        activations = [X]
        pre_activations = []
        dropout_masks: List[Optional[np.ndarray]] = []

        for index, layer in enumerate(self.layers):
            z = np.dot(activations[-1], layer["weights"]) + layer["bias"]
            pre_activations.append(z)

            if index < len(self.layers) - 1:
                a = self._activation(z)
                mask = None
                if training and self.dropout > 0:
                    mask = (np.random.rand(*a.shape) >= self.dropout).astype(np.float32) / (1 - self.dropout)
                    a = a * mask
                activations.append(a)
                dropout_masks.append(mask)
            else:
                a = self._output_activation(z)
                activations.append(a)
                dropout_masks.append(None)

        return activations, pre_activations, dropout_masks

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
        early_stopping: bool = False,
        patience: int = 10,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
        if y_val is not None:
            y_val = np.asarray(y_val, dtype=np.float32)
        
        self._validate_training_data(X, y, X_val, y_val)

        if batch_size is None:
            batch_size = len(X)

        val_loss_history: List[float] = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            indices = np.random.permutation(len(X))
            epoch_loss = 0.0

            for start in range(0, len(X), batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                activations, _, dropout_masks = self.forward(X_batch, training=True)
                output = activations[-1]

                if output.ndim == 2 and y_batch.ndim == 1:
                    y_batch = y_batch.reshape(-1, 1)

                error = y_batch - output
                batch_loss = np.mean(error**2)
                epoch_loss += batch_loss

                delta = error * self._output_activation_derivative(output)

                for layer_index in reversed(range(len(self.layers))):
                    a_prev = activations[layer_index]
                    w = self.layers[layer_index]["weights"]
                    grad_w = np.dot(a_prev.T, delta) / len(X_batch)
                    grad_b = np.mean(delta, axis=0)

                    if self.l2 > 0:
                        grad_w += self.l2 * w

                    self.layers[layer_index]["weights"] += self.learning_rate * grad_w
                    self.layers[layer_index]["bias"] += self.learning_rate * grad_b

                    if layer_index > 0:
                        delta = np.dot(delta, w.T)
                        mask = dropout_masks[layer_index - 1]
                        if mask is not None:
                            delta *= mask
                        delta = delta * self._activation_derivative(activations[layer_index])

            epoch_loss /= max(1, len(X) // batch_size)
            self.loss_history.append(epoch_loss)

            if X_val is not None and y_val is not None:
                val_output = self.predict(X_val)
                val_error = y_val - val_output
                val_loss = np.mean(val_error**2)
                val_loss_history.append(val_loss)

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
        print(f"\nTraining completed in {history['epochs_trained']} epochs")
        print(f"  Final train loss: {history['final_loss']:.6f}")
        if history["final_val_loss"]:
            print(f"  Final val loss:   {history['final_val_loss']:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        activations, _, _ = self.forward(X, training=False)
        output = activations[-1]
        if self.output_size == 1:
            return np.squeeze(output, axis=-1)
        return output

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        predictions = self.predict(X)
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = np.squeeze(predictions, axis=-1)
        y_flat = np.squeeze(y)
        return self._compute_metrics(y_flat, predictions)

    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        print(f"\nTest Metrics:")
        print(f"  MSE:      {metrics['mse']:.6f}")
        print(f"  MAE:      {metrics['mae']:.6f}")
        print(f"  RMSE:     {metrics['rmse']:.6f}")
        if metrics["accuracy"]:
            print(f"  Accuracy: {metrics['accuracy']:.2%}")

    def save(self, path: str) -> None:
        payload = {
            "input_size": self.input_size,
            "hidden_layers": np.array(self.hidden_layers, dtype=np.int32),
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "l2": self.l2,
            "dropout": self.dropout,
            "n_layers": len(self.layers),
            "loss_history": np.array(self.loss_history, dtype=np.float64),
        }
        for index, layer in enumerate(self.layers):
            payload[f"weights_{index}"] = layer["weights"]
            payload[f"bias_{index}"] = layer["bias"]
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "MLP":
        data = np.load(path, allow_pickle=True)
        hidden_layers = data["hidden_layers"].tolist()
        # Backward compatibility: default to "sigmoid" if not saved
        out_act = str(data["output_activation"]) if "output_activation" in data else "sigmoid"
        mlp = cls(
            input_size=int(data["input_size"]),
            hidden_layers=hidden_layers,
            output_size=int(data["output_size"]),
            learning_rate=float(data["learning_rate"]),
            epochs=int(data["epochs"]),
            activation=str(data["activation"]),
            output_activation=out_act,
            l2=float(data["l2"]),
            dropout=float(data["dropout"]),
        )
        for index in range(int(data["n_layers"])):
            mlp.layers[index]["weights"] = data[f"weights_{index}"]
            mlp.layers[index]["bias"] = data[f"bias_{index}"]
        mlp.loss_history = data["loss_history"].tolist()
        return mlp

    def generate_xor_data(self):
        return generate_xor_data()

    def generate_xnor_data(self):
        return generate_xnor_data()

    def generate_not_data(self):
        return generate_not_data()

    def generate_or_data(self):
        return generate_or_data()

    def generate_nor_data(self):
        return generate_nor_data()

    def generate_and_data(self):
        return generate_and_data()

    def generate_nand_data(self):
        return generate_nand_data()


def generate_xor_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=float)
    return X, y


def generate_xnor_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([1, 0, 0, 1], dtype=float)
    return X, y


def GenerateXORMlp(
    learning_rate: float = 0.1,
    epochs: int = 1000,
    activation: str = "tanh",
    output_activation: str = "sigmoid",
    l2: float = 0.0,
    dropout: float = 0.0,
) -> MLP:
    """Create a minimal MLP configured for XOR learning."""
    return GenerateMLP(
        input_size=2,
        hidden_layers=[2],
        output_size=1,
        learning_rate=learning_rate,
        epochs=epochs,
        activation=activation,
        output_activation=output_activation,
        l2=l2,
        dropout=dropout,
    )


def GenerateMLP(
    input_size: int,
    hidden_layers: Optional[List[int]] = None,
    output_size: int = 1,
    learning_rate: float = 0.01,
    epochs: int = 100,
    activation: str = "relu",
    output_activation: str = "sigmoid",
    l2: float = 0.0,
    dropout: float = 0.0,
) -> MLP:
    """
    Automatically generate and initialize an MLP.

    Args:
        output_activation: Activation for the output layer.
            "sigmoid" for classification, "relu" or "linear" for regression.
    """
    return MLP(
        input_size=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,
        learning_rate=learning_rate,
        epochs=epochs,
        activation=activation,
        output_activation=output_activation,
        l2=l2,
        dropout=dropout,
    )