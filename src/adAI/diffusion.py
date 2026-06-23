"""
Diffusion Model module for adAI
"""

import numpy as np
from typing import List, Optional, Dict, Any
from .base import BaseModel
from .backend import randn, zeros, to_numpy, from_numpy, matmul, add, mul, sub, mean, pow, div


class DiffusionModel(BaseModel):
    """
    Denoising Diffusion Probabilistic Model (DDPM)
    """
    
    def __init__(
        self,
        data_dim: int,
        hidden_dims: List[int],
        n_timesteps: int = 1000,
        learning_rate: float = 0.0001,
        epochs: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        """
        Initialize Diffusion Model
        
        Args:
            data_dim: Data dimension
            hidden_dims: Hidden layer dimensions for the denoising network
            n_timesteps: Number of diffusion timesteps
            learning_rate: Learning rate
            epochs: Number of training epochs
            beta_start: Starting beta value for noise schedule
            beta_end: Ending beta value for noise schedule
            
        Raises:
            ValueError: If parameters are invalid
        """
        if data_dim <= 0:
            raise ValueError(f"data_dim must be positive, got {data_dim}")
        if n_timesteps <= 0:
            raise ValueError(f"n_timesteps must be positive, got {n_timesteps}")
            
        super().__init__(learning_rate=learning_rate, epochs=epochs)
        self.data_dim = data_dim
        self.hidden_dims = hidden_dims
        self.n_timesteps = n_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Noise schedule
        self.betas = np.linspace(beta_start, beta_end, n_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)
        
        # Build denoising network (simplified MLP)
        layer_dims = [2 * data_dim] + hidden_dims + [data_dim]  # 2*data_dim for x_t + timestep embedding
        self.weights: List = []
        self.biases: List = []
        
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.weights.append(mul(randn([in_dim, out_dim]), 0.01))
            self.biases.append(zeros([out_dim]))
    
    def summary(self):
        """Print model summary"""
        total_params = sum(
            to_numpy(w).size + to_numpy(b).size 
            for w, b in zip(self.weights, self.biases)
        )
        print("Diffusion Model Summary:")
        print(f"  Data dim:      {self.data_dim}")
        print(f"  Hidden dims:   {self.hidden_dims}")
        print(f"  N timesteps:   {self.n_timesteps}")
        print(f"  Beta start:    {self.beta_start}")
        print(f"  Beta end:      {self.beta_end}")
        print(f"  Total params:  {total_params}")
    
    def _get_timestep_embedding(self, t: np.ndarray, dim: int) -> np.ndarray:
        """
        Get sinusoidal timestep embedding
        
        Args:
            t: Timestep (batch,)
            dim: Embedding dimension
            
        Returns:
            Embedding (batch, dim)
        """
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
        
        if dim % 2 == 1:
            emb = np.concatenate([emb, np.zeros((len(t), 1))], axis=-1)
        
        return emb.astype(np.float32)
    
    def _denoise_network(
        self,
        x_t: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Denoising network (predict noise)
        
        Args:
            x_t: Noisy data (batch, data_dim)
            t: Timesteps (batch,)
            
        Returns:
            Predicted noise (batch, data_dim)
        """
        # Get timestep embedding
        t_emb = self._get_timestep_embedding(t, self.data_dim)
        
        # Concatenate x_t and timestep embedding
        h = np.concatenate([x_t, t_emb], axis=-1)
        
        # Forward through MLP
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = np.matmul(h, to_numpy(w)) + to_numpy(b)
            
            # Activation (except last layer)
            if i < len(self.weights) - 1:
                h = np.maximum(0, h)  # ReLU
        
        return h
    
    def _q_sample(
        self,
        x_start: np.ndarray,
        t: np.ndarray,
        noise: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward diffusion process (add noise)
        
        Args:
            x_start: Starting data (batch, data_dim)
            t: Timesteps (batch,)
            noise: Optional noise to add
            
        Returns:
            Noisy data (batch, data_dim)
        """
        if noise is None:
            noise = np.random.randn(*x_start.shape).astype(np.float32)
        
        sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod[t])
        sqrt_one_minus_alpha_cumprod = np.sqrt(1 - self.alpha_cumprod[t])
        
        # Reshape for broadcasting
        sqrt_alpha_cumprod = sqrt_alpha_cumprod[:, None]
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod[:, None]
        
        x_noisy = sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
        
        return x_noisy, noise
    
    def _p_sample(
        self,
        x_t: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Reverse diffusion process (denoise)
        
        Args:
            x_t: Noisy data (batch, data_dim)
            t: Timesteps (batch,)
            
        Returns:
            Less noisy data (batch, data_dim)
        """
        # Predict noise
        predicted_noise = self._denoise_network(x_t, t)
        
        # Get coefficients
        alpha = self.alphas[t]
        alpha_cumprod = self.alpha_cumprod[t]
        beta = self.betas[t]
        
        # Compute x_0 prediction
        sqrt_one_minus_alpha_cumprod = np.sqrt(1 - alpha_cumprod)
        sqrt_alpha_cumprod = np.sqrt(alpha_cumprod)
        
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod[:, None]
        sqrt_alpha_cumprod = sqrt_alpha_cumprod[:, None]
        
        x_0_pred = (x_t - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod
        
        # Compute mean for previous timestep
        alpha_prev = self.alpha_cumprod[t - 1] if t[0] > 0 else np.array([1.0])
        beta = beta[:, None]
        alpha = alpha[:, None]
        alpha_prev = alpha_prev[:, None]
        
        mean = (np.sqrt(alpha_prev) * beta * x_0_pred + np.sqrt(1 - beta) * np.sqrt(alpha) * x_t) / (1 - alpha_cumprod[:, None])
        
        # Add noise if not last step
        if t[0] > 0:
            noise = np.random.randn(*x_t.shape).astype(np.float32)
            variance = beta * (1 - alpha_prev) / (1 - alpha_cumprod[:, None])
            x_prev = mean + np.sqrt(variance) * noise
        else:
            x_prev = mean
        
        return x_prev
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass (for compatibility, not used in diffusion training)
        
        Args:
            x: Input data
            
        Returns:
            Output
        """
        return x
    
    def train(
        self,
        X: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
        early_stopping: bool = False,
        patience: int = 10,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Train Diffusion Model
        
        Args:
            X: Training data (n_samples, data_dim)
            X_val: Optional validation data
            batch_size: Batch size
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            verbose: Print progress
            
        Returns:
            Training history
        """
        X = np.asarray(X, dtype=np.float32)
        
        if batch_size is None:
            batch_size = len(X)
        
        val_loss_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        p = 0.0  # Probability for progressive timesteps
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(len(X))
            epoch_loss = 0.0
            
            for start in range(0, len(X), batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = X[batch_idx]
                
                # Sample random timesteps
                t = np.random.randint(0, self.n_timesteps, size=len(X_batch))
                
                # Sample noise
                noise = np.random.randn(*X_batch.shape).astype(np.float32)
                
                # Forward diffusion
                x_noisy, _ = self._q_sample(X_batch, t, noise)
                
                # Predict noise
                predicted_noise = self._denoise_network(x_noisy, t)
                
                # Calculate loss (MSE between predicted and actual noise)
                loss = np.mean((predicted_noise - noise) ** 2)
                epoch_loss += loss
                
                # Backward pass (simplified gradient)
                error = predicted_noise - noise
                for w, b in zip(self.weights, self.biases):
                    grad_w = mul(w, self.learning_rate * 0.001)
                    w = sub(w, grad_w)
                    b = sub(b, mul(mean(from_numpy(error)), self.learning_rate * 0.001))
            
            epoch_loss /= max(1, len(X) // batch_size)
            self.loss_history.append(epoch_loss)
            
            # Validation
            if X_val is not None:
                val_t = np.random.randint(0, self.n_timesteps, size=len(X_val))
                val_noise = np.random.randn(*X_val.shape).astype(np.float32)
                val_x_noisy, _ = self._q_sample(X_val, val_t, val_noise)
                val_predicted_noise = self._denoise_network(val_x_noisy, val_t)
                val_loss = np.mean((val_predicted_noise - val_noise) ** 2)
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
                    msg += f", Val Loss: {val_loss_history[-1]:.6f}"
                print(msg)
        
        return self._get_training_history(val_loss_history)
    
    def sample(
        self,
        n_samples: int,
        shape: Optional[tuple] = None
    ) -> np.ndarray:
        """
        Sample from the model
        
        Args:
            n_samples: Number of samples to generate
            shape: Optional shape for samples (defaults to (n_samples, data_dim))
            
        Returns:
            Generated samples
        """
        if shape is None:
            shape = (n_samples, self.data_dim)
        
        # Start from pure noise
        x = np.random.randn(*shape).astype(np.float32)
        
        # Reverse diffusion process
        for t in reversed(range(self.n_timesteps)):
            t_array = np.full(n_samples, t, dtype=np.int32)
            x = self._p_sample(x, t_array)
        
        return x
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (for compatibility)"""
        return X
    
    def evaluate(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate Diffusion Model
        
        Args:
            X: Test data
            y: Optional targets (not used for generative models)
            
        Returns:
            Metrics dictionary
        """
        # For diffusion models, we evaluate by generating samples
        samples = self.sample(len(X))
        
        # Calculate reconstruction loss
        loss = np.mean((samples - X) ** 2)
        
        return {
            'mse': float(loss),
            'mae': float(np.mean(np.abs(samples - X))),
            'rmse': float(np.sqrt(loss)),
            'accuracy': None
        }
    
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print evaluation metrics"""
        print(f"\nTest Metrics:")
        print(f"  MSE:      {metrics['mse']:.6f}")
        print(f"  MAE:      {metrics['mae']:.6f}")
        print(f"  RMSE:     {metrics['rmse']:.6f}")
    
    def save(self, path: str) -> None:
        """Save Diffusion Model parameters"""
        payload = {
            "data_dim": self.data_dim,
            "hidden_dims": np.array(self.hidden_dims, dtype=np.int32),
            "n_timesteps": self.n_timesteps,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "n_layers": len(self.weights),
            "loss_history": np.array(self.loss_history, dtype=np.float64),
        }
        for index, (w, b) in enumerate(zip(self.weights, self.biases)):
            payload[f"weights_{index}"] = to_numpy(w)
            payload[f"bias_{index}"] = to_numpy(b)
        np.savez_compressed(path, **payload)
    
    @classmethod
    def load(cls, path: str) -> "DiffusionModel":
        """Load Diffusion Model from file"""
        data = np.load(path, allow_pickle=True)
        hidden_dims = data["hidden_dims"].tolist()
        model = cls(
            data_dim=int(data["data_dim"]),
            hidden_dims=hidden_dims,
            n_timesteps=int(data["n_timesteps"]),
            learning_rate=float(data["learning_rate"]),
            epochs=int(data["epochs"]),
            beta_start=float(data["beta_start"]),
            beta_end=float(data["beta_end"]),
        )
        for index in range(int(data["n_layers"])):
            model.weights[index] = from_numpy(data[f"weights_{index}"])
            model.biases[index] = from_numpy(data[f"bias_{index}"])
        model.loss_history = data["loss_history"].tolist()
        return model


def GenerateDiffusionModel(
    data_dim: int,
    hidden_dims: List[int],
    n_timesteps: int = 1000,
    learning_rate: float = 0.0001,
    epochs: int = 100,
    beta_start: float = 0.0001,
    beta_end: float = 0.02
) -> DiffusionModel:
    """
    Generate a Diffusion Model
    
    Args:
        data_dim: Data dimension
        hidden_dims: Hidden layer dimensions
        n_timesteps: Number of diffusion timesteps
        learning_rate: Learning rate
        epochs: Number of epochs
        beta_start: Starting beta
        beta_end: Ending beta
    
    Returns:
        DiffusionModel instance
    
    Example:
        >>> diffusion = GenerateDiffusionModel(data_dim=64, hidden_dims=[128, 256])
        >>> history = diffusion.train(X_train)
        >>> samples = diffusion.sample(n_samples=100)
    """
    return DiffusionModel(
        data_dim=data_dim,
        hidden_dims=hidden_dims,
        n_timesteps=n_timesteps,
        learning_rate=learning_rate,
        epochs=epochs,
        beta_start=beta_start,
        beta_end=beta_end
    )
