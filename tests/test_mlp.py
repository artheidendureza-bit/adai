"""
Unit tests for the MLP module of adAI
"""

import pytest
import numpy as np
from adAI import GenerateMLP, MLP, generate_xor_data


def test_generate_mlp_creation():
    """Test that GenerateMLP creates an MLP with expected parameters."""
    mlp = GenerateMLP(
        input_size=4,
        hidden_layers=[8, 16],
        output_size=1,
        activation="relu"
    )
    
    assert mlp is not None
    assert mlp.input_size == 4
    assert mlp.hidden_layers == [8, 16]
    assert mlp.output_size == 1
    assert mlp.activation == "relu"
    assert len(mlp.layers) == 3  # 2 hidden + 1 output


def test_mlp_invalid_input_size():
    """Test that MLP raises error for invalid input_size."""
    with pytest.raises(ValueError):
        GenerateMLP(input_size=0, hidden_layers=[8], output_size=1)


def test_mlp_invalid_output_size():
    """Test that MLP raises error for invalid output_size."""
    with pytest.raises(ValueError):
        GenerateMLP(input_size=4, hidden_layers=[8], output_size=0)


def test_mlp_invalid_hidden_size():
    """Test that MLP raises error for invalid hidden layer size."""
    with pytest.raises(ValueError):
        GenerateMLP(input_size=4, hidden_layers=[0], output_size=1)


def test_mlp_forward():
    """Test that MLP forward pass works correctly."""
    mlp = GenerateMLP(input_size=4, hidden_layers=[8], output_size=1)
    X = np.random.randn(10, 4).astype(np.float32)
    
    output = mlp.forward(X)
    
    assert output.shape == (10, 1)


def test_mlp_forward_single_sample():
    """Test that MLP forward pass works with single sample."""
    mlp = GenerateMLP(input_size=4, hidden_layers=[8], output_size=1)
    X = np.random.randn(4).astype(np.float32)
    
    output = mlp.forward(X)
    
    assert output.shape == (1, 1)


def test_mlp_train():
    """Test that MLP training works correctly."""
    mlp = GenerateMLP(
        input_size=4,
        hidden_layers=[8],
        output_size=1,
        epochs=5,
        learning_rate=0.01
    )
    
    X = np.random.randn(50, 4).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(float)
    
    history = mlp.train(X, y, verbose=False)
    
    assert 'loss_history' in history
    assert len(history['loss_history']) == 5
    assert history['epochs_trained'] == 5


def test_mlp_train_with_validation():
    """Test that MLP training with validation works correctly."""
    mlp = GenerateMLP(
        input_size=4,
        hidden_layers=[8],
        output_size=1,
        epochs=5
    )
    
    X_train = np.random.randn(50, 4).astype(np.float32)
    y_train = (X_train.sum(axis=1) > 0).astype(float)
    X_val = np.random.randn(10, 4).astype(np.float32)
    y_val = (X_val.sum(axis=1) > 0).astype(float)
    
    history = mlp.train(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)
    
    assert 'val_loss_history' in history
    assert len(history['val_loss_history']) == 5


def test_mlp_early_stopping():
    """Test that MLP early stopping works correctly."""
    mlp = GenerateMLP(
        input_size=4,
        hidden_layers=[8],
        output_size=1,
        epochs=100
    )
    
    X_train = np.random.randn(50, 4).astype(np.float32)
    y_train = (X_train.sum(axis=1) > 0).astype(float)
    X_val = np.random.randn(10, 4).astype(np.float32)
    y_val = (X_val.sum(axis=1) > 0).astype(float)
    
    history = mlp.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        early_stopping=True,
        patience=3,
        verbose=False
    )
    
    # Should stop early due to patience
    assert history['epochs_trained'] < 100


def test_mlp_predict():
    """Test that MLP prediction works correctly."""
    mlp = GenerateMLP(input_size=4, hidden_layers=[8], output_size=1)
    X = np.random.randn(10, 4).astype(np.float32)
    
    predictions = mlp.predict(X)
    
    assert predictions.shape == (10,)


def test_mlp_predict_single_sample():
    """Test that MLP prediction works with single sample."""
    mlp = GenerateMLP(input_size=4, hidden_layers=[8], output_size=1)
    X = np.random.randn(4).astype(np.float32)
    
    predictions = mlp.predict(X)
    
    assert predictions.shape == ()


def test_mlp_evaluate():
    """Test that MLP evaluation works correctly."""
    mlp = GenerateMLP(input_size=4, hidden_layers=[8], output_size=1)
    X = np.random.randn(20, 4).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(float)
    
    metrics = mlp.evaluate(X, y)
    
    assert 'mse' in metrics
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert metrics['mse'] >= 0


def test_mlp_save_load():
    """Test that MLP save and load works correctly."""
    mlp = GenerateMLP(
        input_size=4,
        hidden_layers=[8],
        output_size=1,
        epochs=10
    )
    
    X = np.random.randn(20, 4).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(float)
    mlp.train(X, y, verbose=False)
    
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
        temp_path = f.name
    
    try:
        mlp.save(temp_path)
        loaded_mlp = MLP.load(temp_path)
        
        assert loaded_mlp.input_size == mlp.input_size
        assert loaded_mlp.hidden_layers == mlp.hidden_layers
        assert loaded_mlp.output_size == mlp.output_size
        assert len(loaded_mlp.loss_history) == len(mlp.loss_history)
    finally:
        import time
        time.sleep(0.1)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def test_mlp_summary():
    """Test that MLP summary prints without error."""
    mlp = GenerateMLP(input_size=4, hidden_layers=[8], output_size=1)
    
    # Should not raise any exception
    mlp.summary()


def test_mlp_different_activations():
    """Test that MLP works with different activation functions."""
    for activation in ["relu", "sigmoid", "tanh", "linear"]:
        mlp = GenerateMLP(
            input_size=4,
            hidden_layers=[8],
            output_size=1,
            activation=activation
        )
        
        X = np.random.randn(10, 4).astype(np.float32)
        output = mlp.forward(X)
        
        assert output.shape == (10, 1)


def test_mlp_no_hidden_layers():
    """Test that MLP works with no hidden layers."""
    mlp = GenerateMLP(input_size=4, hidden_layers=[], output_size=1)
    
    assert mlp.hidden_layers == []
    assert len(mlp.layers) == 1  # Only output layer
    
    X = np.random.randn(10, 4).astype(np.float32)
    output = mlp.forward(X)
    
    assert output.shape == (10, 1)


def test_mlp_multiple_hidden_layers():
    """Test that MLP works with multiple hidden layers."""
    mlp = GenerateMLP(
        input_size=4,
        hidden_layers=[8, 16, 32],
        output_size=1
    )
    
    assert len(mlp.layers) == 4  # 3 hidden + 1 output
    
    X = np.random.randn(10, 4).astype(np.float32)
    output = mlp.forward(X)
    
    assert output.shape == (10, 1)


def test_mlp_multi_output():
    """Test that MLP works with multiple outputs."""
    mlp = GenerateMLP(input_size=4, hidden_layers=[8], output_size=3)
    
    X = np.random.randn(10, 4).astype(np.float32)
    output = mlp.forward(X)
    
    assert output.shape == (10, 3)


def test_generate_xor_mlp():
    """Test that GenerateXORMlp creates an MLP for XOR."""
    mlp = GenerateXORMlp()
    
    assert mlp.input_size == 2
    assert mlp.output_size == 1
    assert len(mlp.layers) >= 2  # At least hidden + output


def test_generate_xor_data():
    """Test that generate_xor_data works correctly."""
    X, y = generate_xor_data(n_samples=100)
    
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert len(np.unique(y)) == 2  # Binary classification


def test_mlp_xor_training():
    """Test that MLP can learn XOR."""
    mlp = GenerateXORMlp(epochs=50)
    X, y = generate_xor_data(n_samples=100)
    
    history = mlp.train(X, y, verbose=False)
    
    assert 'loss_history' in history
    assert len(history['loss_history']) == 50


def test_mlp_custom_optimizer():
    """Test that MLP works with custom optimizer."""
    mlp = GenerateMLP(
        input_size=4,
        hidden_layers=[8],
        output_size=1,
        optimizer="adam"
    )
    
    X = np.random.randn(10, 4).astype(np.float32)
    output = mlp.forward(X)
    
    assert output.shape == (10, 1)


def test_mlp_custom_learning_rate():
    """Test that MLP works with custom learning rate."""
    mlp = GenerateMLP(
        input_size=4,
        hidden_layers=[8],
        output_size=1,
        learning_rate=0.05
    )
    
    assert mlp.learning_rate == 0.05


def test_mlp_batch_training():
    """Test that MLP batch training works correctly."""
    mlp = GenerateMLP(
        input_size=4,
        hidden_layers=[8],
        output_size=1,
        epochs=5
    )
    
    X = np.random.randn(50, 4).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(float)
    
    history = mlp.train(X, y, batch_size=16, verbose=False)
    
    assert 'loss_history' in history
    assert len(history['loss_history']) == 5
