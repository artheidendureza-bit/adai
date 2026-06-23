"""
Graph Neural Network (GNN) module for adAI
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from .base import BaseModel
from .backend import randn, zeros, to_numpy, from_numpy, matmul, add, mul, sub, mean, pow, div


class Graph:
    """Graph data structure for GNNs"""
    
    def __init__(
        self,
        node_features: np.ndarray,
        adjacency_matrix: np.ndarray,
        edge_features: Optional[np.ndarray] = None
    ):
        """
        Initialize a graph
        
        Args:
            node_features: Node feature matrix (n_nodes, n_features)
            adjacency_matrix: Adjacency matrix (n_nodes, n_nodes)
            edge_features: Optional edge feature matrix (n_edges, n_edge_features)
            
        Raises:
            ValueError: If dimensions are invalid
        """
        node_features = np.asarray(node_features, dtype=np.float32)
        adjacency_matrix = np.asarray(adjacency_matrix, dtype=np.float32)
        
        if node_features.ndim != 2:
            raise ValueError("node_features must be 2D")
        if adjacency_matrix.ndim != 2:
            raise ValueError("adjacency_matrix must be 2D")
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("adjacency_matrix must be square")
        if adjacency_matrix.shape[0] != node_features.shape[0]:
            raise ValueError("adjacency_matrix and node_features must have matching node counts")
            
        self.node_features = node_features
        self.adjacency_matrix = adjacency_matrix
        self.edge_features = edge_features
        self.n_nodes = node_features.shape[0]
        self.n_node_features = node_features.shape[1]
    
    def to(self, device):
        """Move graph to device (placeholder for future device support)"""
        return self


class GCNLayer:
    """Graph Convolutional Network Layer"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "relu"
    ):
        """
        Initialize GCN layer
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            activation: Activation function
        """
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Initialize weights
        self.weights = mul(randn([input_dim, output_dim]), 0.01)
        self.bias = zeros([output_dim])
    
    def forward(self, graph: Graph) -> Graph:
        """
        Forward pass through GCN layer
        
        Args:
            graph: Input graph
            
        Returns:
            Graph with updated node features
        """
        A = graph.adjacency_matrix
        X = graph.node_features
        
        # Add self-loops
        A = A + np.eye(A.shape[0], dtype=np.float32)
        
        # Normalize adjacency matrix (D^(-1/2) * A * D^(-1/2))
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
        D_inv_sqrt = np.diag(D_inv_sqrt)
        A_normalized = from_numpy(D_inv_sqrt @ A @ D_inv_sqrt)
        
        # Graph convolution: A_normalized * X * W + b
        X_tensor = from_numpy(X)
        h = add(matmul(matmul(A_normalized, X_tensor), self.weights), self.bias)
        
        # Apply activation
        if self.activation == "relu":
            h_np = to_numpy(h)
            h_np = np.maximum(0, h_np)
            h = from_numpy(h_np)
        elif self.activation == "sigmoid":
            h_np = to_numpy(h)
            h_np = 1 / (1 + np.exp(-h_np))
            h = from_numpy(h_np)
        elif self.activation == "tanh":
            h_np = to_numpy(h)
            h_np = np.tanh(h_np)
            h = from_numpy(h_np)
        
        # Create new graph with updated features
        new_graph = Graph(to_numpy(h), A)
        return new_graph


class GNN(BaseModel):
    """Graph Neural Network for node classification"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        learning_rate: float = 0.01,
        epochs: int = 100,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """
        Initialize GNN
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of classes)
            learning_rate: Learning rate
            epochs: Number of training epochs
            activation: Activation function
            dropout: Dropout rate
            
        Raises:
            ValueError: If parameters are invalid
        """
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0.0, 1.0)")
            
        super().__init__(learning_rate=learning_rate, epochs=epochs)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.dropout = dropout
        
        # Build layers
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        self.layers: List[GCNLayer] = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.layers.append(GCNLayer(in_dim, out_dim, activation))
    
    def summary(self):
        """Print model summary"""
        total_params = sum(
            to_numpy(layer.weights).size + to_numpy(layer.bias).size 
            for layer in self.layers
        )
        print("GNN Summary:")
        print(f"  Input dim:     {self.input_dim}")
        print(f"  Hidden dims:   {self.hidden_dims}")
        print(f"  Output dim:    {self.output_dim}")
        print(f"  Activation:    {self.activation}")
        print(f"  Dropout:        {self.dropout}")
        print(f"  Layers:        {[self.input_dim] + self.hidden_dims + [self.output_dim]}")
        print(f"  Total params:  {total_params}")
    
    def forward(self, graph: Graph, training: bool = False) -> Graph:
        """
        Forward pass through GNN
        
        Args:
            graph: Input graph
            training: Whether in training mode
            
        Returns:
            Graph with node predictions
        """
        current_graph = graph
        
        for i, layer in enumerate(self.layers):
            current_graph = layer.forward(current_graph)
            
            # Apply dropout (except on last layer)
            if training and self.dropout > 0 and i < len(self.layers) - 1:
                features_np = current_graph.node_features
                mask = (np.random.rand(*features_np.shape) >= self.dropout).astype(np.float32) / (1 - self.dropout)
                current_graph.node_features = features_np * mask
        
        return current_graph
    
    def train(
        self,
        graph: Graph,
        labels: np.ndarray,
        mask: Optional[np.ndarray] = None,
        val_graph: Optional[Graph] = None,
        val_labels: Optional[np.ndarray] = None,
        val_mask: Optional[np.ndarray] = None,
        early_stopping: bool = False,
        patience: int = 10,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Train GNN
        
        Args:
            graph: Training graph
            labels: Node labels (n_nodes,)
            mask: Optional mask for labeled nodes
            val_graph: Optional validation graph
            val_labels: Optional validation labels
            val_mask: Optional validation mask
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            verbose: Print progress
            
        Returns:
            Training history
        """
        labels = np.asarray(labels, dtype=np.float32)
        if mask is None:
            mask = np.ones(graph.n_nodes, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
        
        val_loss_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Forward pass
            output_graph = self.forward(graph, training=True)
            predictions = output_graph.node_features
            
            # Calculate loss (cross-entropy for classification)
            labeled_preds = predictions[mask]
            labeled_labels = labels[mask]
            
            # Softmax
            exp_preds = np.exp(labeled_preds - np.max(labeled_preds, axis=1, keepdims=True))
            softmax_preds = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
            
            # Cross-entropy loss
            nll = -np.log(softmax_preds[np.arange(len(labeled_labels)), labeled_labels.astype(int)])
            loss = np.mean(nll)
            
            self.loss_history.append(loss)
            
            # Backward pass (simplified gradient)
            error = softmax_preds - np.eye(softmax_preds.shape[1])[labeled_labels.astype(int)]
            error = error / len(labeled_labels)
            
            # Update weights (simplified)
            for layer in reversed(self.layers):
                grad_w = mul(layer.weights, self.learning_rate * 0.01)
                layer.weights = sub(layer.weights, grad_w)
                layer.bias = sub(layer.bias, mul(mean(from_numpy(error)), self.learning_rate * 0.01))
            
            # Validation
            if val_graph is not None and val_labels is not None:
                val_output = self.forward(val_graph, training=False)
                val_preds = val_output.node_features
                val_mask_arr = val_mask if val_mask is not None else np.ones(val_graph.n_nodes, dtype=bool)
                
                val_labeled_preds = val_preds[val_mask_arr]
                val_labeled_labels = val_labels[val_mask_arr]
                
                val_exp_preds = np.exp(val_labeled_preds - np.max(val_labeled_preds, axis=1, keepdims=True))
                val_softmax_preds = val_exp_preds / np.sum(val_exp_preds, axis=1, keepdims=True)
                val_nll = -np.log(val_softmax_preds[np.arange(len(val_labeled_labels)), val_labeled_labels.astype(int)])
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
                msg = f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.6f}"
                if val_graph is not None:
                    msg += f", Val Loss: {val_loss_history[-1]:.6f}"
                print(msg)
        
        return self._get_training_history(val_loss_history)
    
    def predict(self, graph: Graph) -> np.ndarray:
        """Make predictions on graph"""
        output_graph = self.forward(graph, training=False)
        predictions = output_graph.node_features
        
        # Apply softmax
        exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        softmax_preds = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        
        return np.argmax(softmax_preds, axis=1)
    
    def evaluate(
        self,
        graph: Graph,
        labels: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate GNN
        
        Args:
            graph: Test graph
            labels: True labels
            mask: Optional mask for evaluation
            
        Returns:
            Metrics dictionary
        """
        predictions = self.predict(graph)
        
        if mask is None:
            mask = np.ones(graph.n_nodes, dtype=bool)
        
        accuracy = np.mean(predictions[mask] == labels[mask])
        
        return {
            'accuracy': float(accuracy),
            'mse': 0.0,  # Placeholder for compatibility
            'mae': 0.0,  # Placeholder for compatibility
            'rmse': 0.0  # Placeholder for compatibility
        }
    
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print evaluation metrics"""
        print(f"\nTest Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
    
    def save(self, path: str) -> None:
        """Save GNN parameters"""
        payload = {
            "input_dim": self.input_dim,
            "hidden_dims": np.array(self.hidden_dims, dtype=np.int32),
            "output_dim": self.output_dim,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "activation": self.activation,
            "dropout": self.dropout,
            "n_layers": len(self.layers),
            "loss_history": np.array(self.loss_history, dtype=np.float64),
        }
        for index, layer in enumerate(self.layers):
            payload[f"weights_{index}"] = to_numpy(layer.weights)
            payload[f"bias_{index}"] = to_numpy(layer.bias)
        np.savez_compressed(path, **payload)
    
    @classmethod
    def load(cls, path: str) -> "GNN":
        """Load GNN from file"""
        data = np.load(path, allow_pickle=True)
        hidden_dims = data["hidden_dims"].tolist()
        gnn = cls(
            input_dim=int(data["input_dim"]),
            hidden_dims=hidden_dims,
            output_dim=int(data["output_dim"]),
            learning_rate=float(data["learning_rate"]),
            epochs=int(data["epochs"]),
            activation=str(data["activation"]),
            dropout=float(data["dropout"]),
        )
        for index in range(int(data["n_layers"])):
            gnn.layers[index].weights = from_numpy(data[f"weights_{index}"])
            gnn.layers[index].bias = from_numpy(data[f"bias_{index}"])
        gnn.loss_history = data["loss_history"].tolist()
        return gnn


def GenerateGNN(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    learning_rate: float = 0.01,
    epochs: int = 100,
    activation: str = "relu",
    dropout: float = 0.0
) -> GNN:
    """
    Generate a GNN
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        learning_rate: Learning rate
        epochs: Number of epochs
        activation: Activation function
        dropout: Dropout rate
    
    Returns:
        GNN instance
    
    Example:
        >>> gnn = GenerateGNN(input_dim=16, hidden_dims=[32, 16], output_dim=4)
        >>> history = gnn.train(graph, labels, mask=mask)
        >>> predictions = gnn.predict(graph)
    """
    return GNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        learning_rate=learning_rate,
        epochs=epochs,
        activation=activation,
        dropout=dropout
    )
