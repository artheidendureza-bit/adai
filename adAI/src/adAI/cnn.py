"""
CNN module for adAI
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .base import BaseModel
from .activations import get_activation, softmax, sigmoid, relu


class Conv2DLayer:
    """A 2D Convolutional layer with sliding patch dot products."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: str = "same"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # He Normal initialization
        limit = np.sqrt(2.0 / (kernel_size * kernel_size * in_channels))
        self.W = (np.random.randn(kernel_size, kernel_size, in_channels, out_channels) * limit).astype(np.float32)
        self.b = np.zeros(out_channels, dtype=np.float32)
        
        # Caches for backpropagation
        self.X_in = None
        self.X_pad = None
        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X_in = X
        B, H, W, C_in = X.shape
        K = self.kernel_size
        p = (K - 1) // 2 if self.padding == "same" else 0
        H_out = H if self.padding == "same" else H - K + 1
        W_out = W if self.padding == "same" else W - K + 1
        
        self.X_pad = np.pad(X, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant')
        Z = np.zeros((B, H_out, W_out, self.out_channels), dtype=np.float32)
        
        # Vectorized convolution using im2col approach
        # Extract patches and reshape for matrix multiplication
        patches = []
        for i in range(K):
            for j in range(K):
                slice_X = self.X_pad[:, i:i+H_out, j:j+W_out, :]
                patches.append(slice_X.reshape(B, H_out * W_out, C_in))
        
        # Stack patches and reshape for efficient computation
        patches_stacked = np.stack(patches, axis=0)  # (K, K, B, H_out*W_out, C_in)
        W_reshaped = self.W.reshape(K * K, C_in, self.out_channels)
        patches_reshaped = patches_stacked.reshape(K * K, B * H_out * W_out, C_in)
        
        # Compute convolution in one operation
        conv_result = np.einsum('kbc,kco->kbo', patches_reshaped, W_reshaped)
        conv_result = conv_result.reshape(K, K, B, H_out, W_out, self.out_channels)
        Z = np.sum(conv_result, axis=(0, 1))
        
        Z += self.b
        return Z

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        B, H_out, W_out, C_out = dZ.shape
        K = self.kernel_size
        p = (K - 1) // 2 if self.padding == "same" else 0
        
        self.dW = np.zeros_like(self.W)
        self.db = np.sum(dZ, axis=(0, 1, 2)) / B
        
        dX_pad = np.zeros_like(self.X_pad)
        
        for i in range(K):
            for j in range(K):
                slice_X = self.X_pad[:, i:i+H_out, j:j+W_out, :]
                # Compute dW for this kernel pixel offset
                self.dW[i, j, :, :] = np.tensordot(slice_X, dZ, axes=([0, 1, 2], [0, 1, 2])) / B
                
                # Compute gradient for padded input
                W_pixel = self.W[i, j, :, :]
                dX_pad[:, i:i+H_out, j:j+W_out, :] += np.dot(dZ, W_pixel.T)
                
        if p > 0:
            dX = dX_pad[:, p:-p, p:-p, :]
        else:
            dX = dX_pad
        return dX


class MaxPool2DLayer:
    """A 2D MaxPooling layer with spatial pooling mask tracking."""
    
    def __init__(self, pool_size: int = 2):
        self.pool_size = pool_size
        self.X_in = None
        self.mask = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X_in = X
        B, H, W, C = X.shape
        s = self.pool_size
        H_out = H // s
        W_out = W // s
        Z = np.zeros((B, H_out, W_out, C), dtype=np.float32)
        self.mask = np.zeros_like(X, dtype=bool)
        
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * s
                h_end = h_start + s
                w_start = w * s
                w_end = w_start + s
                
                patch = X[:, h_start:h_end, w_start:w_end, :]
                max_val = np.max(patch, axis=(1, 2))
                Z[:, h, w, :] = max_val
                
                max_val_expanded = max_val[:, np.newaxis, np.newaxis, :]
                self.mask[:, h_start:h_end, w_start:w_end, :] = (patch == max_val_expanded)
        return Z

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        B, H_out, W_out, C = dZ.shape
        s = self.pool_size
        dX = np.zeros_like(self.X_in)
        
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * s
                h_end = h_start + s
                w_start = w * s
                w_end = w_start + s
                
                dZ_val = dZ[:, h:h+1, w:w+1, :]
                patch_mask = self.mask[:, h_start:h_end, w_start:w_end, :]
                dX[:, h_start:h_end, w_start:w_end, :] += patch_mask * dZ_val
        return dX


class FlattenLayer:
    """A layer to flatten 4D feature maps to 2D dense vector batches."""
    
    def __init__(self):
        self.in_shape = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.in_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        return dZ.reshape(self.in_shape)


class DenseLayer:
    """A fully-connected dense linear projection layer."""
    
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        
        limit = np.sqrt(2.0 / in_features)
        self.W = (np.random.randn(in_features, out_features) * limit).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        
        self.X_in = None
        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X_in = X
        return np.dot(X, self.W) + self.b

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        B = dZ.shape[0]
        self.dW = np.dot(self.X_in.T, dZ) / B
        self.db = np.mean(dZ, axis=0)
        return np.dot(dZ, self.W.T)


class ReLULayer:
    """Element-wise Rectified Linear Unit activation layer."""
    
    def __init__(self):
        self.X_in = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X_in = X
        return relu(X)

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        return dZ * (self.X_in > 0).astype(np.float32)


class CNN(BaseModel):
    """A Convolutional Neural Network class supporting Conv2D, MaxPooling, Flatten, and Dense layers."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        conv_layers: List[Tuple[int, int]],
        dense_layers: List[int],
        output_size: int,
        pool_size: int = 2,
        learning_rate: float = 0.01,
        epochs: int = 100,
        output_activation: Optional[str] = None,
        l2: float = 0.0,
        optimizer: str = "adam",
    ):
        """
        Initialize CNN
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            conv_layers: List of (filters, kernel_size) tuples for conv layers
            dense_layers: List of dense layer sizes
            output_size: Number of output units
            pool_size: Pooling size for max pooling
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            output_activation: Output activation function
            l2: L2 regularization coefficient
            optimizer: Optimizer to use ("sgd", "adam", "adamw", "radam", "radamw")
            
        Raises:
            ValueError: If parameters are invalid
        """
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (height, width, channels), got {input_shape}")
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        if optimizer.lower() not in ["sgd", "adam", "adamw", "radam", "radamw"]:
            raise ValueError(f"Unknown optimizer: {optimizer}")
            
        super().__init__(learning_rate=learning_rate, epochs=epochs)
        self.input_shape = input_shape
        self.conv_layers = conv_layers
        self.pool_size = pool_size
        self.dense_layers = dense_layers
        self.output_size = output_size
        self.l2 = l2
        self.optimizer = optimizer.lower()
        self.t = 0

        if output_activation is None:
            self.output_activation = "sigmoid" if output_size == 1 else "softmax"
        else:
            self.output_activation = output_activation
            
        # Get output activation function
        if self.output_activation == "softmax":
            self._output_activation_fn = softmax
        elif self.output_activation == "sigmoid":
            self._output_activation_fn = sigmoid
        elif self.output_activation == "relu":
            self._output_activation_fn = relu
        else:
            self._output_activation_fn = lambda x: x  # linear

        # Dynamically build layers
        self.layers: List[object] = []
        
        # 1. Convolutional Block
        in_c = self.input_shape[2]
        for filters, kernel in self.conv_layers:
            self.layers.append(Conv2DLayer(in_c, filters, kernel, padding="same"))
            self.layers.append(ReLULayer())
            if self.pool_size > 1:
                self.layers.append(MaxPool2DLayer(self.pool_size))
            in_c = filters

        # 2. Flatten
        self.layers.append(FlattenLayer())

        # Determine input dimension for dense block using a dummy forward pass
        dummy_in = np.zeros((1, *self.input_shape), dtype=np.float32)
        out = dummy_in
        for layer in self.layers:
            out = layer.forward(out)
        flat_size = out.shape[1]

        # 3. Dense Block
        in_features = flat_size
        for dense_size in self.dense_layers:
            self.layers.append(DenseLayer(in_features, dense_size))
            self.layers.append(ReLULayer())
            in_features = dense_size

        # 4. Output Dense Layer
        self.layers.append(DenseLayer(in_features, self.output_size))

        self.loss_history: List[float] = []

    def summary(self):
        """Print CNN structural summary."""
        total_params = 0
        print("CNN Summary:")
        print(f"  Input shape:        {self.input_shape}")
        print(f"  Conv layers:        {self.conv_layers}")
        print(f"  Pool size:          {self.pool_size}")
        print(f"  Dense layers:       {self.dense_layers}")
        print(f"  Output size:        {self.output_size}")
        print(f"  Output activation:  {self.output_activation}")
        print(f"  L2 regularizer:     {self.l2}")
        print("  Layer structure:")
        
        for idx, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            if hasattr(layer, "W") and hasattr(layer, "b"):
                params = layer.W.size + layer.b.size
                total_params += params
                print(f"    [{idx:02d}] {layer_name:<18} | Param count: {params:,} (W: {layer.W.shape}, b: {layer.b.shape})")
            else:
                print(f"    [{idx:02d}] {layer_name:<18} | Param count: 0")
        print(f"  Total params:       {total_params:,}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the CNN layers.
        
        Args:
            X: Input data
            
        Returns:
            Features before output activation
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _apply_output_activation(self, X: np.ndarray) -> np.ndarray:
        return self._output_activation_fn(X)

    def _update_parameters(self):
        self.t += 1
        lr = self.learning_rate
        wd = self.l2
        opt = self.optimizer.lower()
        
        # Hyperparameters for adaptive optimizers
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        # Precompute RAdam values if needed
        if opt in ("radam", "radamw"):
            rho_inf = 2.0 / (1.0 - beta2) - 1.0
            rho_t = rho_inf - 2.0 * self.t * (beta2 ** self.t) / (1.0 - beta2 ** self.t)
            
        for layer in self.layers:
            if hasattr(layer, "W") and hasattr(layer, "b"):
                # Allocate moment states on demand
                if not hasattr(layer, "m_W") or layer.m_W is None:
                    layer.m_W = np.zeros_like(layer.W)
                    layer.v_W = np.zeros_like(layer.W)
                    layer.m_b = np.zeros_like(layer.b)
                    layer.v_b = np.zeros_like(layer.b)
                
                # Update weights (W) and biases (b)
                for param, grad, m, v in [
                    (layer.W, layer.dW, layer.m_W, layer.v_W),
                    (layer.b, layer.db, layer.m_b, layer.v_b)
                ]:
                    is_weight = (param is layer.W)
                    
                    if opt in ("sgd", "adam", "radam"):
                        if is_weight and wd > 0:
                            grad_wd = grad - wd * param
                        else:
                            grad_wd = grad
                    else:
                        grad_wd = grad

                    if opt == "sgd":
                        param += lr * grad_wd
                        
                    elif opt in ("adam", "adamw"):
                        m *= beta1
                        m += (1 - beta1) * grad_wd
                        
                        v *= beta2
                        v += (1 - beta2) * (grad_wd ** 2)
                        
                        m_hat = m / (1.0 - beta1 ** self.t)
                        v_hat = v / (1.0 - beta2 ** self.t)
                        
                        step = m_hat / (np.sqrt(v_hat) + eps)
                        
                        if opt == "adamw" and is_weight and wd > 0:
                            param *= (1.0 - lr * wd)
                            
                        param += lr * step
                        
                    elif opt in ("radam", "radamw"):
                        m *= beta1
                        m += (1 - beta1) * grad_wd
                        
                        v *= beta2
                        v += (1 - beta2) * (grad_wd ** 2)
                        
                        m_hat = m / (1.0 - beta1 ** self.t)
                        
                        if rho_t > 4.0:
                            num = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                            den = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t
                            r_t = np.sqrt(num / den)
                            
                            v_hat = np.sqrt(v / (1.0 - beta2 ** self.t))
                            step = r_t * m_hat / (v_hat + eps)
                        else:
                            step = m_hat
                            
                        if opt == "radamw" and is_weight and wd > 0:
                            param *= (1.0 - lr * wd)
                            
                        param += lr * step

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
        
        self._validate_training_data(X, y, X_val, y_val)

        # Standardize target shape
        if self.output_size == 1:
            y = np.squeeze(y)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
        else:
            if y.ndim == 1:
                # Automate one-hot encoding for integer labels
                num_classes = self.output_size
                one_hot = np.zeros((len(y), num_classes), dtype=np.float32)
                one_hot[np.arange(len(y)), y.astype(int)] = 1.0
                y = one_hot

        # Process validation target shape
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)
            if self.output_size == 1:
                y_val = np.squeeze(y_val)
                if y_val.ndim == 1:
                    y_val = y_val.reshape(-1, 1)
            else:
                if y_val.ndim == 1:
                    num_classes = self.output_size
                    one_hot = np.zeros((len(y_val), num_classes), dtype=np.float32)
                    one_hot[np.arange(len(y_val)), y_val.astype(int)] = 1.0
                    y_val = one_hot

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

                # 1. Forward Pass
                out = self.forward(X_batch)
                output = self._apply_output_activation(out)

                if output.ndim == 2 and y_batch.ndim == 1:
                    y_batch = y_batch.reshape(-1, 1)

                error = y_batch - output
                batch_loss = np.mean(error**2)
                epoch_loss += batch_loss

                # 2. Output Layer Delta Calculation (MSE base)
                if self.output_activation == "softmax":
                    delta = output * (error - np.sum(error * output, axis=-1, keepdims=True))
                elif self.output_activation == "sigmoid":
                    delta = error * output * (1 - output)
                elif self.output_activation == "relu":
                    delta = error * (output > 0).astype(np.float32)
                else:  # linear
                    delta = error

                # 3. Backward Pass
                grad = delta
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                # 4. Weight Updates
                self._update_parameters()

            epoch_loss /= max(1, len(X) // batch_size)
            self.loss_history.append(epoch_loss)

            # Validation cycle
            if X_val is not None and y_val is not None:
                val_output = self.predict(X_val)
                if val_output.ndim == 2 and y_val.ndim == 1:
                    y_val = y_val.reshape(-1, 1)
                elif val_output.ndim == 1 and y_val.ndim == 2 and y_val.shape[1] == 1:
                    y_val = np.squeeze(y_val, axis=-1)
                
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
        """Print training results summary."""
        print(f"\nTraining completed in {history['epochs_trained']} epochs")
        print(f"  Final train loss: {history['final_loss']:.6f}")
        if history["final_val_loss"]:
            print(f"  Final val loss:   {history['final_val_loss']:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on inputs."""
        X = np.asarray(X, dtype=np.float32)
        is_single_sample = X.ndim == 3
        if is_single_sample:
            X = X[np.newaxis, ...]
            
        out = self.forward(X)
        output = self._apply_output_activation(out)
        
        if self.output_size == 1:
            output = np.squeeze(output, axis=-1)
            if is_single_sample:
                return output[0]
        else:
            if is_single_sample:
                return output[0]
        return output

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate CNN on test inputs and targets."""
        predictions = self.predict(X)
        
        if self.output_size == 1:
            predictions = np.squeeze(predictions)
            y_flat = np.squeeze(y)
            mse = np.mean((y_flat - predictions) ** 2)
            mae = np.mean(np.abs(y_flat - predictions))
            binary_predictions = (predictions > 0.5).astype(int)
            accuracy = np.mean(binary_predictions == y_flat) if np.all((y_flat == 0) | (y_flat == 1)) else None
        else:
            if y.ndim == 1:
                true_labels = y.astype(int)
            else:
                true_labels = np.argmax(y, axis=1)
            pred_labels = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_labels == true_labels)
            
            if y.ndim > 1:
                mse = np.mean((y - predictions) ** 2)
                mae = np.mean(np.abs(y - predictions))
            else:
                num_classes = self.output_size
                one_hot = np.zeros((len(y), num_classes), dtype=np.float32)
                one_hot[np.arange(len(y)), y.astype(int)] = 1.0
                mse = np.mean((one_hot - predictions) ** 2)
                mae = np.mean(np.abs(one_hot - predictions))
                
        return {
            "mse": mse,
            "mae": mae,
            "rmse": np.sqrt(mse),
            "accuracy": accuracy,
        }

    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print evaluation metrics summary."""
        print(f"\nTest Metrics:")
        print(f"  MSE:      {metrics['mse']:.6f}")
        print(f"  MAE:      {metrics['mae']:.6f}")
        print(f"  RMSE:     {metrics['rmse']:.6f}")
        if metrics["accuracy"] is not None:
            print(f"  Accuracy: {metrics['accuracy']:.2%}")

    def save(self, path: str) -> None:
        """Save CNN state and architecture to compressed NPZ format."""
        payload = {
            "input_shape": np.array(self.input_shape, dtype=np.int32),
            "conv_layers_filters": np.array([filters for filters, _ in self.conv_layers], dtype=np.int32),
            "conv_layers_kernels": np.array([kernel for _, kernel in self.conv_layers], dtype=np.int32),
            "pool_size": self.pool_size,
            "dense_layers": np.array(self.dense_layers, dtype=np.int32),
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "output_activation": self.output_activation,
            "l2": self.l2,
            "optimizer": self.optimizer,
            "t": self.t,
            "loss_history": np.array(self.loss_history, dtype=np.float64),
        }
        
        learnable_index = 0
        for layer in self.layers:
            if hasattr(layer, "W") and hasattr(layer, "b"):
                payload[f"weights_{learnable_index}"] = layer.W
                payload[f"bias_{learnable_index}"] = layer.b
                
                # Save optimizer moment states if they exist
                if hasattr(layer, "m_W") and layer.m_W is not None:
                    payload[f"m_W_{learnable_index}"] = layer.m_W
                    payload[f"v_W_{learnable_index}"] = layer.v_W
                    payload[f"m_b_{learnable_index}"] = layer.m_b
                    payload[f"v_b_{learnable_index}"] = layer.v_b
                    
                learnable_index += 1
                
        payload["n_learnable_layers"] = learnable_index
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "CNN":
        """Load CNN state and architecture from compressed NPZ format."""
        data = np.load(path, allow_pickle=True)
        
        input_shape = tuple(data["input_shape"].tolist())
        filters = data["conv_layers_filters"].tolist()
        kernels = data["conv_layers_kernels"].tolist()
        conv_layers = list(zip(filters, kernels))
        
        pool_size = int(data["pool_size"])
        dense_layers = data["dense_layers"].tolist()
        output_size = int(data["output_size"])
        
        opt = str(data["optimizer"]) if "optimizer" in data else "sgd"
        t_val = int(data["t"]) if "t" in data else 0
        
        cnn = cls(
            input_shape=input_shape,
            conv_layers=conv_layers,
            pool_size=pool_size,
            dense_layers=dense_layers,
            output_size=output_size,
            learning_rate=float(data["learning_rate"]),
            epochs=int(data["epochs"]),
            output_activation=str(data["output_activation"]),
            l2=float(data["l2"]),
            optimizer=opt,
        )
        
        cnn.t = t_val
        
        learnable_index = 0
        for layer in cnn.layers:
            if hasattr(layer, "W") and hasattr(layer, "b"):
                layer.W = data[f"weights_{learnable_index}"]
                layer.b = data[f"bias_{learnable_index}"]
                
                # Restore optimizer moment states if they were saved
                if f"m_W_{learnable_index}" in data:
                    layer.m_W = data[f"m_W_{learnable_index}"]
                    layer.v_W = data[f"v_W_{learnable_index}"]
                    layer.m_b = data[f"m_b_{learnable_index}"]
                    layer.v_b = data[f"v_b_{learnable_index}"]
                else:
                    layer.m_W = None
                    layer.v_W = None
                    layer.m_b = None
                    layer.v_b = None
                    
                learnable_index += 1
                
        cnn.loss_history = data["loss_history"].tolist()
        return cnn


def GenerateCNN(
    input_shape: Tuple[int, int, int],
    conv_layers: List[Tuple[int, int]],
    dense_layers: List[int],
    output_size: int,
    pool_size: int = 2,
    learning_rate: float = 0.01,
    epochs: int = 100,
    output_activation: Optional[str] = None,
    l2: float = 0.0,
    optimizer: str = "adam",
) -> CNN:
    """Automatically generate and initialize a CNN."""
    return CNN(
        input_shape=input_shape,
        conv_layers=conv_layers,
        dense_layers=dense_layers,
        output_size=output_size,
        pool_size=pool_size,
        learning_rate=learning_rate,
        epochs=epochs,
        output_activation=output_activation,
        l2=l2,
        optimizer=optimizer,
    )
