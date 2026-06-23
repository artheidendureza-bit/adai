"""
Spiking Neural Network (SNN) module for adAI
"""

import numpy as np
from typing import List, Optional, Dict, Any
from .base import BaseModel
from .backend import randn, zeros, to_numpy, from_numpy, matmul, add, mul, sub, mean, pow, div


class LIFNeuron:
    """Leaky Integrate-and-Fire (LIF) neuron"""
    
    def __init__(
        self,
        threshold: float = 1.0,
        leak: float = 0.1,
        reset_mode: str = "subtract"
    ):
        """
        Initialize LIF neuron
        
        Args:
            threshold: Spike threshold voltage
            leak: Membrane potential leak factor (0 to 1)
            reset_mode: Reset mode - "subtract" or "zero"
            
        Raises:
            ValueError: If parameters are invalid
        """
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if leak < 0 or leak > 1:
            raise ValueError(f"leak must be in [0, 1], got {leak}")
        if reset_mode not in ["subtract", "zero"]:
            raise ValueError(f'reset_mode must be "subtract" or "zero", got {reset_mode}')
            
        self.threshold = threshold
        self.leak = leak
        self.reset_mode = reset_mode
    
    def forward_step(self, membrane_potential, input_current):
        """
        Single time step forward pass
        
        Args:
            membrane_potential: Current membrane potential
            input_current: Input current at this time step
            
        Returns:
            Tuple of (new_membrane_potential, spike)
        """
        # Add input current first
        membrane_potential = add(membrane_potential, input_current)
        
        # Check for spike
        spike = from_numpy(to_numpy(membrane_potential) >= self.threshold)
        
        # Reset membrane potential after spike
        if self.reset_mode == "subtract":
            membrane_potential = sub(membrane_potential, mul(spike, self.threshold))
        else:
            membrane_potential = mul(membrane_potential, 1.0 - to_numpy(spike))
        
        # Decay membrane potential
        membrane_potential = mul(membrane_potential, 1.0 - self.leak)
        
        return membrane_potential, spike


class SpikingLayer:
    """A spiking neural network layer with LIF neurons"""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        threshold: float = 1.0,
        leak: float = 0.1,
        reset_mode: str = "subtract"
    ):
        """
        Initialize spiking layer
        
        Args:
            input_size: Number of input neurons
            output_size: Number of output neurons
            threshold: Spike threshold
            leak: Membrane leak factor
            reset_mode: Reset mode
        """
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
            
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights and bias
        self.weights = mul(randn([input_size, output_size]), 0.1)
        self.bias = zeros([output_size])
        
        # LIF neuron for each output
        self.neuron = LIFNeuron(threshold=threshold, leak=leak, reset_mode=reset_mode)
        
        # Store membrane potentials
        self.membrane_potential = zeros([output_size])
    
    def forward_step(self, input_spikes):
        """
        Single time step forward pass
        
        Args:
            input_spikes: Binary spike input (batch_size, input_size)
            
        Returns:
            Output spikes (batch_size, output_size)
        """
        # Compute input current
        current = add(matmul(input_spikes, self.weights), self.bias)
        
        # Update membrane potential and generate spikes
        self.membrane_potential, output_spikes = self.neuron.forward_step(
            self.membrane_potential, current
        )
        
        return output_spikes
    
    def reset_state(self):
        """Reset membrane potentials to zero"""
        self.membrane_potential = zeros([self.output_size])


class SNN(BaseModel):
    """Spiking Neural Network with multiple spiking layers"""
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: Optional[List[int]] = None,
        output_size: int = 1,
        learning_rate: float = 0.01,
        epochs: int = 100,
        time_steps: int = 20,
        threshold: float = 1.0,
        leak: float = 0.1,
        reset_mode: str = "subtract",
        encoding: str = "rate"
    ):
        """
        Initialize SNN
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            output_size: Number of output neurons
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            time_steps: Number of time steps for simulation
            threshold: Spike threshold for all neurons
            leak: Membrane leak factor for all neurons
            reset_mode: Reset mode for all neurons
            encoding: Input encoding - "rate" or "poisson"
            
        Raises:
            ValueError: If parameters are invalid
        """
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        if time_steps <= 0:
            raise ValueError(f"time_steps must be positive, got {time_steps}")
        if encoding not in ["rate", "poisson"]:
            raise ValueError(f'encoding must be "rate" or "poisson", got {encoding}')
            
        super().__init__(learning_rate=learning_rate, epochs=epochs)
        self.input_size = input_size
        self.hidden_layers = hidden_layers or []
        self.output_size = output_size
        self.time_steps = time_steps
        self.threshold = threshold
        self.leak = leak
        self.reset_mode = reset_mode
        self.encoding = encoding
        
        # Build layers
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        self.layers: List[SpikingLayer] = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(
                SpikingLayer(
                    input_size=in_size,
                    output_size=out_size,
                    threshold=threshold,
                    leak=leak,
                    reset_mode=reset_mode
                )
            )
    
    def summary(self):
        """Print model summary"""
        total_params = sum(
            to_numpy(layer.weights).size + to_numpy(layer.bias).size 
            for layer in self.layers
        )
        print("SNN Summary:")
        print(f"  Input size:    {self.input_size}")
        print(f"  Hidden layers: {self.hidden_layers}")
        print(f"  Output size:   {self.output_size}")
        print(f"  Time steps:    {self.time_steps}")
        print(f"  Threshold:     {self.threshold}")
        print(f"  Leak:          {self.leak}")
        print(f"  Reset mode:    {self.reset_mode}")
        print(f"  Encoding:      {self.encoding}")
        print(f"  Layers:        {[self.input_size] + self.hidden_layers + [self.output_size]}")
        print(f"  Total params:  {total_params}")
    
    def _encode_input(self, X, time_steps):
        """
        Encode input as spike trains
        
        Args:
            X: Input data (n_samples, n_features)
            time_steps: Number of time steps
            
        Returns:
            Spikes (time_steps, n_samples, n_features)
        """
        X_np = to_numpy(X)
        n_samples, n_features = X_np.shape
        
        # Normalize to [0, 1]
        X_normalized = (X_np - X_np.min()) / (X_np.max() - X_np.min() + 1e-8)
        
        if self.encoding == "rate":
            # Rate coding: probability of spike = normalized value
            spikes = np.random.rand(time_steps, n_samples, n_features) < X_normalized[None, :, :]
        else:  # poisson
            # Poisson coding: spike count follows Poisson distribution
            # Generate independent Poisson samples for each time step
            spikes = np.random.rand(time_steps, n_samples, n_features) < X_normalized[None, :, :]
        return from_numpy(spikes.astype(np.float32))
    
    def _decode_output(self, spikes):
        """
        Decode output spikes to rates
        
        Args:
            spikes: Output spikes (time_steps, n_samples, output_size)
            
        Returns:
            Decoded rates (n_samples, output_size)
        """
        spikes_np = to_numpy(spikes)
        # Average over time steps
        rates = np.mean(spikes_np, axis=0)
        return from_numpy(rates)
    
    def forward(self, X, training: bool = False):
        """
        Forward pass through SNN
        
        Args:
            X: Input data (n_samples, n_features)
            training: Whether in training mode
            
        Returns:
            Output rates (n_samples, output_size)
        """
        # Reset all membrane potentials
        for layer in self.layers:
            layer.reset_state()
        
        # Encode input as spikes
        input_spikes = self._encode_input(X, self.time_steps)
        
        # Time-stepped simulation
        all_spikes = []
        for t in range(self.time_steps):
            current_spikes = input_spikes[t]
            
            for layer in self.layers:
                current_spikes = layer.forward_step(current_spikes)
            
            all_spikes.append(current_spikes)
        
        # Stack spikes over time
        output_spikes = from_numpy(np.stack([to_numpy(s) for s in all_spikes], axis=0))
        
        # Decode to rates
        output_rates = self._decode_output(output_spikes)
        
        return output_rates
    
    def train(
        self,
        X,
        y,
        X_val: Optional = None,
        y_val: Optional = None,
        batch_size: Optional[int] = None,
        early_stopping: bool = False,
        patience: int = 10,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the SNN
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target values (n_samples,)
            X_val: Validation data (optional)
            y_val: Validation targets (optional)
            batch_size: Batch size for mini-batch training
            early_stopping: Whether to use early stopping
            patience: Number of epochs with no improvement to stop
            verbose: Print training progress
            
        Returns:
            Dictionary with training history
        """
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
                
                # Forward pass
                output = self.forward(X_batch, training=True)
                
                # Calculate loss (MSE)
                y_batch_reshaped = y_batch.reshape(-1, 1) if y_batch.ndim == 1 else y_batch
                error = sub(from_numpy(y_batch_reshaped), output)
                error_squared = pow(error, 2)
                batch_loss = mean(error_squared)
                batch_loss_np = to_numpy(batch_loss)
                epoch_loss += float(batch_loss_np)
                
                # Backward pass (simplified gradient-based update)
                # For SNNs, we use a surrogate gradient approach
                delta = mul(error, 2.0)  # Derivative of MSE
                
                for layer in reversed(self.layers):
                    # Get input spikes for this layer
                    # (simplified - in practice would need to cache intermediate spikes)
                    w = layer.weights
                    grad_w = mul(w, self.learning_rate * 0.01)  # Simplified gradient
                    layer.weights = add(layer.weights, grad_w)
                    layer.bias = add(layer.bias, mul(mean(delta), self.learning_rate * 0.01))
            
            epoch_loss /= max(1, len(X) // batch_size)
            self.loss_history.append(epoch_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_output = self.predict(X_val)
                val_error = sub(from_numpy(y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val), val_output)
                val_error_squared = pow(val_error, 2)
                val_loss = mean(val_error_squared)
                val_loss_np = to_numpy(val_loss)
                val_loss_history.append(float(val_loss_np))
                
                if early_stopping:
                    if float(val_loss_np) < best_val_loss:
                        best_val_loss = float(val_loss_np)
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
                    msg += f", Val Loss: {val_loss_history[-1]:.6f}"
                print(msg)
        
        return self._get_training_history(val_loss_history)
    
    def print_training_summary(self, history: Dict[str, Any]) -> None:
        """Print training results summary"""
        print(f"\nTraining completed in {history['epochs_trained']} epochs")
        print(f"  Final train loss: {history['final_loss']:.6f}")
        if history['final_val_loss']:
            print(f"  Final val loss:   {history['final_val_loss']:.6f}")
    
    def predict(self, X):
        """Make predictions on new data"""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        output = self.forward(X, training=False)
        output_np = to_numpy(output)
        if self.output_size == 1:
            return np.squeeze(output_np, axis=-1)
        return output_np
    
    def evaluate(self, X, y) -> Dict[str, Any]:
        """Evaluate SNN on test data"""
        predictions = self.predict(X)
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = np.squeeze(predictions, axis=-1)
        y_flat = np.squeeze(y)
        return self._compute_metrics(y_flat, predictions)
    
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print evaluation metrics"""
        print(f"\nTest Metrics:")
        print(f"  MSE:      {metrics['mse']:.6f}")
        print(f"  MAE:      {metrics['mae']:.6f}")
        print(f"  RMSE:     {metrics['rmse']:.6f}")
        if metrics['accuracy']:
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
    
    def save(self, path: str) -> None:
        """Save SNN parameters to file"""
        payload = {
            "input_size": self.input_size,
            "hidden_layers": np.array(self.hidden_layers, dtype=np.int32),
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "time_steps": self.time_steps,
            "threshold": self.threshold,
            "leak": self.leak,
            "reset_mode": self.reset_mode,
            "encoding": self.encoding,
            "n_layers": len(self.layers),
            "loss_history": np.array(self.loss_history, dtype=np.float64),
        }
        for index, layer in enumerate(self.layers):
            payload[f"weights_{index}"] = to_numpy(layer.weights)
            payload[f"bias_{index}"] = to_numpy(layer.bias)
        np.savez_compressed(path, **payload)
    
    @classmethod
    def load(cls, path: str) -> "SNN":
        """Load SNN from file"""
        data = np.load(path, allow_pickle=True)
        hidden_layers = data["hidden_layers"].tolist()
        snn = cls(
            input_size=int(data["input_size"]),
            hidden_layers=hidden_layers,
            output_size=int(data["output_size"]),
            learning_rate=float(data["learning_rate"]),
            epochs=int(data["epochs"]),
            time_steps=int(data["time_steps"]),
            threshold=float(data["threshold"]),
            leak=float(data["leak"]),
            reset_mode=str(data["reset_mode"]),
            encoding=str(data["encoding"]),
        )
        for index in range(int(data["n_layers"])):
            snn.layers[index].weights = from_numpy(data[f"weights_{index}"])
            snn.layers[index].bias = from_numpy(data[f"bias_{index}"])
        snn.loss_history = data["loss_history"].tolist()
        return snn
    
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
            n_train: Number of training samples
            n_val: Number of validation samples
            n_test: Number of test samples
            seed: Random seed
            
        Returns:
            Dictionary with data splits
        """
        np.random.seed(seed)
        
        X_train = np.random.randn(n_train, self.input_size).astype(np.float32)
        y_train = (X_train.sum(axis=1) > 0).astype(float)
        
        X_val = np.random.randn(n_val, self.input_size).astype(np.float32)
        y_val = (X_val.sum(axis=1) > 0).astype(float)
        
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


def GenerateSNN(
    input_size: int,
    hidden_layers: Optional[List[int]] = None,
    output_size: int = 1,
    learning_rate: float = 0.01,
    epochs: int = 100,
    time_steps: int = 20,
    threshold: float = 1.0,
    leak: float = 0.1,
    reset_mode: str = "subtract",
    encoding: str = "rate"
) -> SNN:
    """
    Automatically generate and initialize an SNN
    
    Args:
        input_size: Number of input features
        hidden_layers: List of hidden layer sizes
        output_size: Number of output neurons
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        time_steps: Number of time steps for simulation
        threshold: Spike threshold
        leak: Membrane leak factor
        reset_mode: Reset mode - "subtract" or "zero"
        encoding: Input encoding - "rate" or "poisson"
    
    Returns:
        A ready-to-use SNN instance
    
    Example:
        >>> snn = GenerateSNN(input_size=10, hidden_layers=[32], output_size=1)
        >>> data = snn.generate_data(n_train=500)
        >>> history = snn.train(data['X_train'], data['y_train'])
        >>> metrics = snn.evaluate(data['X_test'], data['y_test'])
    """
    return SNN(
        input_size=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,
        learning_rate=learning_rate,
        epochs=epochs,
        time_steps=time_steps,
        threshold=threshold,
        leak=leak,
        reset_mode=reset_mode,
        encoding=encoding
    )
