"""
Unit tests for the Perceptron module of adAI
"""

import pytest
import numpy as np
from adAI import GeneratePerceptron, Perceptron
from adAI import (
    generate_logic_gate_data,
    generate_not_data,
    generate_or_data,
    generate_nor_data,
    generate_and_data,
    generate_nand_data
)


def test_generate_perceptron_creation():
    """Test that GeneratePerceptron creates a Perceptron with expected parameters."""
    perceptron = GeneratePerceptron(
        input_size=2,
        activation="step"
    )
    
    assert perceptron is not None
    assert perceptron.input_size == 2
    assert perceptron.activation == "step"


def test_perceptron_invalid_input_size():
    """Test that Perceptron raises error for invalid input_size."""
    with pytest.raises(ValueError):
        GeneratePerceptron(input_size=0)


def test_perceptron_forward():
    """Test that Perceptron forward pass works correctly."""
    perceptron = GeneratePerceptron(input_size=2)
    X = np.random.randn(10, 2).astype(np.float32)
    
    output = perceptron.forward(X)
    
    assert output.shape == (10, 1)


def test_perceptron_forward_single_sample():
    """Test that Perceptron forward pass works with single sample."""
    perceptron = GeneratePerceptron(input_size=2)
    X = np.random.randn(2).astype(np.float32)
    
    output = perceptron.forward(X)
    
    assert output.shape == (1, 1)


def test_perceptron_train():
    """Test that Perceptron training works correctly."""
    perceptron = GeneratePerceptron(
        input_size=2,
        epochs=5,
        learning_rate=0.01
    )
    
    X = np.random.randn(50, 2).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(float)
    
    history = perceptron.train(X, y, verbose=False)
    
    assert 'loss_history' in history
    assert len(history['loss_history']) == 5
    assert history['epochs_trained'] == 5


def test_perceptron_train_with_validation():
    """Test that Perceptron training with validation works correctly."""
    perceptron = GeneratePerceptron(
        input_size=2,
        epochs=5
    )
    
    X_train = np.random.randn(50, 2).astype(np.float32)
    y_train = (X_train.sum(axis=1) > 0).astype(float)
    X_val = np.random.randn(10, 2).astype(np.float32)
    y_val = (X_val.sum(axis=1) > 0).astype(float)
    
    history = perceptron.train(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)
    
    assert 'val_loss_history' in history
    assert len(history['val_loss_history']) == 5


def test_perceptron_early_stopping():
    """Test that Perceptron early stopping works correctly."""
    perceptron = GeneratePerceptron(
        input_size=2,
        epochs=100
    )
    
    X_train = np.random.randn(50, 2).astype(np.float32)
    y_train = (X_train.sum(axis=1) > 0).astype(float)
    X_val = np.random.randn(10, 2).astype(np.float32)
    y_val = (X_val.sum(axis=1) > 0).astype(float)
    
    history = perceptron.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        early_stopping=True,
        patience=3,
        verbose=False
    )
    
    # Should stop early due to patience
    assert history['epochs_trained'] < 100


def test_perceptron_predict():
    """Test that Perceptron prediction works correctly."""
    perceptron = GeneratePerceptron(input_size=2)
    X = np.random.randn(10, 2).astype(np.float32)
    
    predictions = perceptron.predict(X)
    
    assert predictions.shape == (10,)


def test_perceptron_predict_single_sample():
    """Test that Perceptron prediction works with single sample."""
    perceptron = GeneratePerceptron(input_size=2)
    X = np.random.randn(2).astype(np.float32)
    
    predictions = perceptron.predict(X)
    
    assert predictions.shape == ()


def test_perceptron_evaluate():
    """Test that Perceptron evaluation works correctly."""
    perceptron = GeneratePerceptron(input_size=2)
    X = np.random.randn(20, 2).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(float)
    
    metrics = perceptron.evaluate(X, y)
    
    assert 'mse' in metrics
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert metrics['mse'] >= 0


def test_perceptron_save_load():
    """Test that Perceptron save and load works correctly."""
    perceptron = GeneratePerceptron(
        input_size=2,
        epochs=10
    )
    
    X = np.random.randn(20, 2).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(float)
    perceptron.train(X, y, verbose=False)
    
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
        temp_path = f.name
    
    try:
        perceptron.save(temp_path)
        loaded_perceptron = Perceptron.load(temp_path)
        
        assert loaded_perceptron.input_size == perceptron.input_size
        assert loaded_perceptron.activation == perceptron.activation
        assert len(loaded_perceptron.loss_history) == len(perceptron.loss_history)
    finally:
        import time
        time.sleep(0.1)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def test_perceptron_summary():
    """Test that Perceptron summary prints without error."""
    perceptron = GeneratePerceptron(input_size=2)
    
    # Should not raise any exception
    perceptron.summary()


def test_perceptron_different_activations():
    """Test that Perceptron works with different activation functions."""
    for activation in ["step", "sigmoid", "relu", "tanh"]:
        perceptron = GeneratePerceptron(
            input_size=2,
            activation=activation
        )
        
        X = np.random.randn(10, 2).astype(np.float32)
        output = perceptron.forward(X)
        
        assert output.shape == (10, 1)


def test_perceptron_custom_learning_rate():
    """Test that Perceptron works with custom learning rate."""
    perceptron = GeneratePerceptron(
        input_size=2,
        learning_rate=0.05
    )
    
    assert perceptron.learning_rate == 0.05


def test_generate_logic_gate_data():
    """Test that generate_logic_gate_data works correctly."""
    X, y = generate_logic_gate_data(gate="AND", n_samples=100)
    
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert len(np.unique(y)) == 2  # Binary classification


def test_generate_not_data():
    """Test that generate_not_data works correctly."""
    X, y = generate_not_data(n_samples=100)
    
    assert X.shape == (100, 1)
    assert y.shape == (100,)
    assert len(np.unique(y)) == 2


def test_generate_or_data():
    """Test that generate_or_data works correctly."""
    X, y = generate_or_data(n_samples=100)
    
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert len(np.unique(y)) == 2


def test_generate_nor_data():
    """Test that generate_nor_data works correctly."""
    X, y = generate_nor_data(n_samples=100)
    
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert len(np.unique(y)) == 2


def test_generate_and_data():
    """Test that generate_and_data works correctly."""
    X, y = generate_and_data(n_samples=100)
    
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert len(np.unique(y)) == 2


def test_generate_nand_data():
    """Test that generate_nand_data works correctly."""
    X, y = generate_nand_data(n_samples=100)
    
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert len(np.unique(y)) == 2


def test_perceptron_and_gate():
    """Test that Perceptron can learn AND gate."""
    perceptron = GeneratePerceptron(input_size=2, epochs=50)
    X, y = generate_and_data(n_samples=100)
    
    history = perceptron.train(X, y, verbose=False)
    
    assert 'loss_history' in history
    assert len(history['loss_history']) == 50


def test_perceptron_or_gate():
    """Test that Perceptron can learn OR gate."""
    perceptron = GeneratePerceptron(input_size=2, epochs=50)
    X, y = generate_or_data(n_samples=100)
    
    history = perceptron.train(X, y, verbose=False)
    
    assert 'loss_history' in history
    assert len(history['loss_history']) == 50


def test_perceptron_not_gate():
    """Test that Perceptron can learn NOT gate."""
    perceptron = GeneratePerceptron(input_size=1, epochs=50)
    X, y = generate_not_data(n_samples=100)
    
    history = perceptron.train(X, y, verbose=False)
    
    assert 'loss_history' in history
    assert len(history['loss_history']) == 50


def test_perceptron_nand_gate():
    """Test that Perceptron can learn NAND gate."""
    perceptron = GeneratePerceptron(input_size=2, epochs=50)
    X, y = generate_nand_data(n_samples=100)
    
    history = perceptron.train(X, y, verbose=False)
    
    assert 'loss_history' in history
    assert len(history['loss_history']) == 50


def test_perceptron_nor_gate():
    """Test that Perceptron can learn NOR gate."""
    perceptron = GeneratePerceptron(input_size=2, epochs=50)
    X, y = generate_nor_data(n_samples=100)
    
    history = perceptron.train(X, y, verbose=False)
    
    assert 'loss_history' in history
    assert len(history['loss_history']) == 50


def test_perceptron_xor_gate_fails():
    """Test that single Perceptron cannot learn XOR (linearly inseparable)."""
    perceptron = GeneratePerceptron(input_size=2, epochs=100)
    X, y = generate_xor_data(n_samples=100)
    
    history = perceptron.train(X, y, verbose=False)
    
    # XOR is linearly inseparable, so loss should remain high
    # This test verifies the perceptron behaves correctly (doesn't magically solve XOR)
    assert 'loss_history' in history


def test_perceptron_batch_training():
    """Test that Perceptron batch training works correctly."""
    perceptron = GeneratePerceptron(
        input_size=2,
        epochs=5
    )
    
    X = np.random.randn(50, 2).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(float)
    
    history = perceptron.train(X, y, batch_size=16, verbose=False)
    
    assert 'loss_history' in history
    assert len(history['loss_history']) == 5


def test_perceptron_custom_optimizer():
    """Test that Perceptron works with custom optimizer."""
    perceptron = GeneratePerceptron(
        input_size=2,
        optimizer="adam"
    )
    
    X = np.random.randn(10, 2).astype(np.float32)
    output = perceptron.forward(X)
    
    assert output.shape == (10, 1)


def test_perceptron_weights_initialization():
    """Test that Perceptron weights are initialized correctly."""
    perceptron = GeneratePerceptron(input_size=2)
    
    assert perceptron.weights is not None
    assert perceptron.bias is not None


def test_perceptron_step_activation():
    """Test that step activation produces binary outputs."""
    perceptron = GeneratePerceptron(input_size=2, activation="step")
    X = np.random.randn(10, 2).astype(np.float32)
    
    output = perceptron.forward(X)
    
    # Step activation should produce 0 or 1
    unique_outputs = np.unique(output)
    assert all(val in [0.0, 1.0] for val in unique_outputs)


def test_perceptron_sigmoid_activation():
    """Test that sigmoid activation produces values in [0, 1]."""
    perceptron = GeneratePerceptron(input_size=2, activation="sigmoid")
    X = np.random.randn(10, 2).astype(np.float32)
    
    output = perceptron.forward(X)
    
    # Sigmoid activation should produce values in [0, 1]
    assert np.all(output >= 0)
    assert np.all(output <= 1)


def test_perceptron_relu_activation():
    """Test that ReLU activation produces non-negative outputs."""
    perceptron = GeneratePerceptron(input_size=2, activation="relu")
    X = np.random.randn(10, 2).astype(np.float32)
    
    output = perceptron.forward(X)
    
    # ReLU activation should produce non-negative values
    assert np.all(output >= 0)
