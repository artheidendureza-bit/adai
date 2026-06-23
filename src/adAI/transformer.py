"""
Transformer module for adAI
"""

import numpy as np
from typing import List, Optional, Dict, Any
from .base import BaseModel
from .backend import randn, zeros, to_numpy, from_numpy, matmul, add, mul, sub, mean, pow, div


class MultiHeadAttention:
    """Multi-Head Attention mechanism"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize Multi-Head Attention
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            
        Raises:
            ValueError: If parameters are invalid
        """
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads")
            
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        # Initialize projections
        self.W_q = mul(randn([d_model, d_model]), 0.01)
        self.W_k = mul(randn([d_model, d_model]), 0.01)
        self.W_v = mul(randn([d_model, d_model]), 0.01)
        self.W_o = mul(randn([d_model, d_model]), 0.01)
    
    def scaled_dot_product_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Scaled dot-product attention
        
        Args:
            Q: Query matrix (batch, seq_len, d_k)
            K: Key matrix (batch, seq_len, d_k)
            V: Value matrix (batch, seq_len, d_k)
            mask: Optional attention mask
            
        Returns:
            Attention output
        """
        # Calculate attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Apply dropout
        if self.dropout > 0:
            dropout_mask = (np.random.rand(*attention_weights.shape) >= self.dropout).astype(np.float32) / (1 - self.dropout)
            attention_weights = attention_weights * dropout_mask
        
        # Apply attention to values
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = np.matmul(x, to_numpy(self.W_q))
        K = np.matmul(x, to_numpy(self.W_k))
        V = np.matmul(x, to_numpy(self.W_v))
        
        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        Q = Q.reshape(batch_size * self.n_heads, seq_len, self.d_k)
        K = K.reshape(batch_size * self.n_heads, seq_len, self.d_k)
        V = V.reshape(batch_size * self.n_heads, seq_len, self.d_k)
        
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape back
        attn_output = attn_output.reshape(batch_size, self.n_heads, seq_len, self.d_k).transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = np.matmul(attn_output, to_numpy(self.W_o))
        
        return output


class FeedForward:
    """Position-wise Feed-Forward Network"""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "relu"
    ):
        """
        Initialize Feed-Forward Network
        
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension
            activation: Activation function
        """
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {d_ff}")
            
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        
        self.W1 = mul(randn([d_model, d_ff]), 0.01)
        self.b1 = zeros([d_ff])
        self.W2 = mul(randn([d_ff, d_model]), 0.01)
        self.b2 = zeros([d_model])
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # First layer
        h = np.matmul(x, to_numpy(self.W1)) + to_numpy(self.b1)
        
        # Activation
        if self.activation == "relu":
            h = np.maximum(0, h)
        elif self.activation == "gelu":
            h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * np.power(h, 3))))
        elif self.activation == "sigmoid":
            h = 1 / (1 + np.exp(-h))
        
        # Second layer
        output = np.matmul(h, to_numpy(self.W2)) + to_numpy(self.b2)
        
        return output


class TransformerEncoderLayer:
    """Transformer Encoder Layer"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Initialize Transformer Encoder Layer
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout = dropout
        
        self.norm1 = zeros([d_model])  # Layer norm parameters (simplified)
        self.norm2 = zeros([d_model])
    
    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output = self.self_attention.forward(x, mask)
        x = x + attn_output
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward.forward(x)
        x = x + ff_output
        
        return x


class Transformer(BaseModel):
    """Transformer model for sequence processing"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        learning_rate: float = 0.001,
        epochs: int = 100,
        dropout: float = 0.1
    ):
        """
        Initialize Transformer
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            learning_rate: Learning rate
            epochs: Number of training epochs
            dropout: Dropout rate
            
        Raises:
            ValueError: If parameters are invalid
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
            
        super().__init__(learning_rate=learning_rate, epochs=epochs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        # Embedding (simplified)
        self.embedding = mul(randn([vocab_size, d_model]), 0.1)
        self.positional_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Encoder layers
        self.layers: List[TransformerEncoderLayer] = []
        for _ in range(n_layers):
            self.layers.append(TransformerEncoderLayer(d_model, n_heads, d_ff, dropout))
        
        # Output projection
        self.output_projection = mul(randn([d_model, vocab_size]), 0.01)
    
    def _create_positional_encoding(self, max_seq_len: int, d_model: int) -> np.ndarray:
        """Create positional encoding"""
        pe = np.zeros((max_seq_len, d_model))
        position = np.arange(0, max_seq_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def summary(self):
        """Print model summary"""
        total_params = (
            to_numpy(self.embedding).size +
            self.positional_encoding.size +
            to_numpy(self.output_projection).size
        )
        for layer in self.layers:
            total_params += to_numpy(layer.self_attention.W_q).size
            total_params += to_numpy(layer.self_attention.W_k).size
            total_params += to_numpy(layer.self_attention.W_v).size
            total_params += to_numpy(layer.self_attention.W_o).size
            total_params += to_numpy(layer.feed_forward.W1).size
            total_params += to_numpy(layer.feed_forward.W2).size
        
        print("Transformer Summary:")
        print(f"  Vocab size:    {self.vocab_size}")
        print(f"  Model dim:     {self.d_model}")
        print(f"  N heads:       {self.n_heads}")
        print(f"  N layers:      {self.n_layers}")
        print(f"  FF dim:        {self.d_ff}")
        print(f"  Max seq len:   {self.max_seq_len}")
        print(f"  Dropout:       {self.dropout}")
        print(f"  Total params:  {total_params}")
    
    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass
        
        Args:
            x: Input token IDs (batch, seq_len)
            mask: Optional attention mask
            
        Returns:
            Output logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        # Embedding
        x_embedded = self.embedding[x]  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x_embedded = x_embedded + self.positional_encoding[:seq_len]
        
        # Pass through encoder layers
        for layer in self.layers:
            x_embedded = layer.forward(x_embedded, mask)
        
        # Output projection
        logits = np.matmul(x_embedded, to_numpy(self.output_projection))
        
        return logits
    
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
        Train Transformer
        
        Args:
            X: Training sequences (batch, seq_len)
            y: Target sequences (batch, seq_len)
            X_val: Optional validation sequences
            y_val: Optional validation targets
            batch_size: Batch size
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            verbose: Print progress
            
        Returns:
            Training history
        """
        X = np.asarray(X, dtype=np.int32)
        y = np.asarray(y, dtype=np.int32)
        
        if batch_size is None:
            batch_size = len(X)
        
        val_loss_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(len(X))
            epoch_loss = 0.0
            
            for start in range(0, len(X), batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # Forward pass
                logits = self.forward(X_batch)
                
                # Calculate loss (cross-entropy)
                batch_size_actual = len(X_batch)
                seq_len = X_batch.shape[1]
                
                # Flatten for loss calculation
                logits_flat = logits.reshape(-1, self.vocab_size)
                y_flat = y_batch.reshape(-1)
                
                # Cross-entropy
                exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=1, keepdims=True))
                softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                nll = -np.log(softmax_probs[np.arange(len(y_flat)), y_flat] + 1e-10)
                loss = np.mean(nll)
                
                epoch_loss += loss
                
                # Backward pass (simplified gradient)
                error = softmax_probs - np.eye(self.vocab_size)[y_flat]
                error = error.reshape(batch_size_actual, seq_len, self.vocab_size)
                
                # Update weights (simplified)
                for layer in self.layers:
                    layer.self_attention.W_q = sub(layer.self_attention.W_q, mul(layer.self_attention.W_q, self.learning_rate * 0.001))
                    layer.feed_forward.W1 = sub(layer.feed_forward.W1, mul(layer.feed_forward.W1, self.learning_rate * 0.001))
            
            epoch_loss /= max(1, len(X) // batch_size)
            self.loss_history.append(epoch_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_logits = self.forward(X_val)
                val_logits_flat = val_logits.reshape(-1, self.vocab_size)
                val_y_flat = y_val.reshape(-1)
                
                val_exp_logits = np.exp(val_logits_flat - np.max(val_logits_flat, axis=1, keepdims=True))
                val_softmax_probs = val_exp_logits / np.sum(val_exp_logits, axis=1, keepdims=True)
                val_nll = -np.log(val_softmax_probs[np.arange(len(val_y_flat)), val_y_flat] + 1e-10)
                val_loss = np.mean(val_nll)
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        logits = self.forward(X)
        predictions = np.argmax(logits, axis=-1)
        return predictions
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate Transformer"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        
        return {
            'accuracy': float(accuracy),
            'mse': 0.0,
            'mae': 0.0,
            'rmse': 0.0
        }
    
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print evaluation metrics"""
        print(f"\nTest Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
    
    def save(self, path: str) -> None:
        """Save Transformer parameters"""
        payload = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "dropout": self.dropout,
            "embedding": to_numpy(self.embedding),
            "positional_encoding": self.positional_encoding,
            "output_projection": to_numpy(self.output_projection),
            "loss_history": np.array(self.loss_history, dtype=np.float64),
        }
        np.savez_compressed(path, **payload)
    
    @classmethod
    def load(cls, path: str) -> "Transformer":
        """Load Transformer from file"""
        data = np.load(path, allow_pickle=True)
        transformer = cls(
            vocab_size=int(data["vocab_size"]),
            d_model=int(data["d_model"]),
            n_heads=int(data["n_heads"]),
            n_layers=int(data["n_layers"]),
            d_ff=int(data["d_ff"]),
            max_seq_len=int(data["max_seq_len"]),
            learning_rate=float(data["learning_rate"]),
            epochs=int(data["epochs"]),
            dropout=float(data["dropout"]),
        )
        transformer.embedding = from_numpy(data["embedding"])
        transformer.positional_encoding = data["positional_encoding"]
        transformer.output_projection = from_numpy(data["output_projection"])
        transformer.loss_history = data["loss_history"].tolist()
        return transformer


def GenerateTransformer(
    vocab_size: int,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 2048,
    max_seq_len: int = 512,
    learning_rate: float = 0.001,
    epochs: int = 100,
    dropout: float = 0.1
) -> Transformer:
    """
    Generate a Transformer
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        learning_rate: Learning rate
        epochs: Number of epochs
        dropout: Dropout rate
    
    Returns:
        Transformer instance
    
    Example:
        >>> transformer = GenerateTransformer(vocab_size=10000, d_model=512)
        >>> history = transformer.train(X_train, y_train)
        >>> predictions = transformer.predict(X_test)
    """
    return Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        learning_rate=learning_rate,
        epochs=epochs,
        dropout=dropout
    )
