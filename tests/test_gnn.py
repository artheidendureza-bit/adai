"""
Unit tests for the GNN module of adAI
"""

import pytest
import numpy as np
from adAI import GenerateGNN, GNN, Graph


def test_graph_creation():
    """Test that Graph is created with correct attributes."""
    node_features = np.random.randn(10, 16).astype(np.float32)
    adjacency_matrix = np.random.randint(0, 2, (10, 10)).astype(np.float32)
    
    graph = Graph(node_features, adjacency_matrix)
    
    assert graph.n_nodes == 10
    assert graph.n_node_features == 16
    assert graph.node_features.shape == (10, 16)
    assert graph.adjacency_matrix.shape == (10, 10)


def test_graph_invalid_dimensions():
    """Test that Graph raises error for invalid dimensions."""
    node_features = np.random.randn(10, 16).astype(np.float32)
    adjacency_matrix = np.random.randint(0, 2, (8, 8)).astype(np.float32)
    
    with pytest.raises(ValueError):
        Graph(node_features, adjacency_matrix)


def test_graph_invalid_adjacency():
    """Test that Graph raises error for non-square adjacency matrix."""
    node_features = np.random.randn(10, 16).astype(np.float32)
    adjacency_matrix = np.random.randint(0, 2, (10, 8)).astype(np.float32)
    
    with pytest.raises(ValueError):
        Graph(node_features, adjacency_matrix)


def test_gcn_layer_creation():
    """Test that GCNLayer is created with correct parameters."""
    from adAI.gnn import GCNLayer
    
    layer = GCNLayer(input_dim=16, output_dim=32, activation="relu")
    
    assert layer.input_dim == 16
    assert layer.output_dim == 32
    assert layer.activation == "relu"


def test_gcn_layer_invalid_input_dim():
    """Test that GCNLayer raises error for invalid input dimension."""
    from adAI.gnn import GCNLayer
    
    with pytest.raises(ValueError):
        GCNLayer(input_dim=0, output_dim=32)


def test_gcn_layer_forward():
    """Test that GCNLayer forward pass works correctly."""
    from adAI.gnn import GCNLayer
    
    node_features = np.random.randn(10, 16).astype(np.float32)
    adjacency_matrix = np.random.randint(0, 2, (10, 10)).astype(np.float32)
    graph = Graph(node_features, adjacency_matrix)
    
    layer = GCNLayer(input_dim=16, output_dim=32, activation="relu")
    output_graph = layer.forward(graph)
    
    assert output_graph.node_features.shape == (10, 32)


def test_generate_gnn_creation():
    """Test that GenerateGNN creates a GNN with expected parameters."""
    gnn = GenerateGNN(
        input_dim=16,
        hidden_dims=[32, 16],
        output_dim=4,
        activation="relu"
    )
    
    assert gnn is not None
    assert gnn.input_dim == 16
    assert gnn.hidden_dims == [32, 16]
    assert gnn.output_dim == 4
    assert gnn.activation == "relu"
    assert len(gnn.layers) == 3  # 2 hidden + 1 output


def test_gnn_invalid_input_dim():
    """Test that GNN raises error for invalid input dimension."""
    with pytest.raises(ValueError):
        GenerateGNN(input_dim=0, hidden_dims=[32], output_dim=4)


def test_gnn_invalid_output_dim():
    """Test that GNN raises error for invalid output dimension."""
    with pytest.raises(ValueError):
        GenerateGNN(input_dim=16, hidden_dims=[32], output_dim=0)


def test_gnn_invalid_dropout():
    """Test that GNN raises error for invalid dropout."""
    with pytest.raises(ValueError):
        GenerateGNN(input_dim=16, hidden_dims=[32], output_dim=4, dropout=1.5)


def test_gnn_forward():
    """Test that GNN forward pass works correctly."""
    node_features = np.random.randn(10, 16).astype(np.float32)
    adjacency_matrix = np.random.randint(0, 2, (10, 10)).astype(np.float32)
    graph = Graph(node_features, adjacency_matrix)
    
    gnn = GenerateGNN(input_dim=16, hidden_dims=[32], output_dim=4)
    output_graph = gnn.forward(graph)
    
    assert output_graph.node_features.shape == (10, 4)


def test_gnn_train():
    """Test that GNN training works correctly."""
    node_features = np.random.randn(10, 16).astype(np.float32)
    adjacency_matrix = np.random.randint(0, 2, (10, 10)).astype(np.float32)
    graph = Graph(node_features, adjacency_matrix)
    
    labels = np.random.randint(0, 4, 10)
    mask = np.array([True] * 5 + [False] * 5)
    
    gnn = GenerateGNN(input_dim=16, hidden_dims=[32], output_dim=4, epochs=5)
    history = gnn.train(graph, labels, mask=mask, verbose=False)
    
    assert 'loss_history' in history
    assert len(history['loss_history']) == 5
    assert history['epochs_trained'] == 5


def test_gnn_predict():
    """Test that GNN prediction works correctly."""
    node_features = np.random.randn(10, 16).astype(np.float32)
    adjacency_matrix = np.random.randint(0, 2, (10, 10)).astype(np.float32)
    graph = Graph(node_features, adjacency_matrix)
    
    gnn = GenerateGNN(input_dim=16, hidden_dims=[32], output_dim=4)
    predictions = gnn.predict(graph)
    
    assert predictions.shape == (10,)
    assert predictions.dtype == np.int64


def test_gnn_evaluate():
    """Test that GNN evaluation works correctly."""
    node_features = np.random.randn(10, 16).astype(np.float32)
    adjacency_matrix = np.random.randint(0, 2, (10, 10)).astype(np.float32)
    graph = Graph(node_features, adjacency_matrix)
    
    labels = np.random.randint(0, 4, 10)
    
    gnn = GenerateGNN(input_dim=16, hidden_dims=[32], output_dim=4)
    metrics = gnn.evaluate(graph, labels)
    
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1


def test_gnn_save_load():
    """Test that GNN save and load works correctly."""
    node_features = np.random.randn(10, 16).astype(np.float32)
    adjacency_matrix = np.random.randint(0, 2, (10, 10)).astype(np.float32)
    graph = Graph(node_features, adjacency_matrix)
    
    labels = np.random.randint(0, 4, 10)
    mask = np.array([True] * 5 + [False] * 5)
    
    gnn = GenerateGNN(input_dim=16, hidden_dims=[32], output_dim=4, epochs=5)
    gnn.train(graph, labels, mask=mask, verbose=False)
    
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
        temp_path = f.name
    
    try:
        gnn.save(temp_path)
        loaded_gnn = GNN.load(temp_path)
        
        assert loaded_gnn.input_dim == gnn.input_dim
        assert loaded_gnn.hidden_dims == gnn.hidden_dims
        assert loaded_gnn.output_dim == gnn.output_dim
        assert len(loaded_gnn.loss_history) == len(gnn.loss_history)
    finally:
        import time
        time.sleep(0.1)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def test_gnn_summary():
    """Test that GNN summary prints without error."""
    gnn = GenerateGNN(input_dim=16, hidden_dims=[32], output_dim=4)
    
    # Should not raise any exception
    gnn.summary()
