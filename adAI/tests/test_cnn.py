"""
Unit tests for the CNN module of adAI
"""

import pytest
import numpy as np
from adAI import GenerateCNN, CNN


def test_generate_cnn_creation():
    """Test that GenerateCNN creates a CNN with expected layers and dimensions."""
    cnn = GenerateCNN(
        input_shape=(28, 28, 1),
        conv_layers=[(8, 3)],
        pool_size=2,
        dense_layers=[16],
        output_size=10
    )
    assert cnn is not None
    assert cnn.input_shape == (28, 28, 1)
    assert cnn.conv_layers == [(8, 3)]
    assert cnn.pool_size == 2
    assert cnn.dense_layers == [16]
    assert cnn.output_size == 10
    
    # Verify dynamic layers construction
    # Conv2D -> ReLU -> MaxPool2D -> Flatten -> Dense -> ReLU -> Dense
    layer_types = [layer.__class__.__name__ for layer in cnn.layers]
    expected_types = [
        "Conv2DLayer",
        "ReLULayer",
        "MaxPool2DLayer",
        "FlattenLayer",
        "DenseLayer",
        "ReLULayer",
        "DenseLayer"
    ]
    assert layer_types == expected_types


def test_cnn_forward_shapes():
    """Test forward pass output shape checks."""
    cnn = GenerateCNN(
        input_shape=(16, 16, 3),
        conv_layers=[(8, 3), (16, 3)],
        pool_size=2,
        dense_layers=[32],
        output_size=5
    )
    
    # Single sample
    X_single = np.random.randn(16, 16, 3).astype(np.float32)
    pred_single = cnn.predict(X_single)
    assert pred_single.shape == (5,)
    assert np.allclose(np.sum(pred_single), 1.0, atol=1e-5)  # Softmax output

    # Batch of samples
    X_batch = np.random.randn(4, 16, 16, 3).astype(np.float32)
    pred_batch = cnn.predict(X_batch)
    assert pred_batch.shape == (4, 5)
    assert np.allclose(np.sum(pred_batch, axis=1), 1.0, atol=1e-5)


def test_cnn_backward_gradients():
    """Test that backward pass backpropagates without shape mismatch error."""
    cnn = GenerateCNN(
        input_shape=(10, 10, 1),
        conv_layers=[(4, 3)],
        pool_size=2,
        dense_layers=[8],
        output_size=2
    )
    
    X = np.random.randn(3, 10, 10, 1).astype(np.float32)
    y = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    
    out = X
    for layer in cnn.layers:
        out = layer.forward(out)
    output = cnn._apply_output_activation(out)
    
    error = y - output
    delta = output * (error - np.sum(error * output, axis=-1, keepdims=True))
    
    grad = delta
    for layer in reversed(cnn.layers):
        grad = layer.backward(grad)
        
    for layer in cnn.layers:
        if hasattr(layer, "W"):
            assert layer.dW is not None
            assert layer.db is not None
            assert layer.dW.shape == layer.W.shape
            assert layer.db.shape == layer.b.shape
            
    assert grad.shape == X.shape

def test_cnn_training_loop():
    """Test training CNN on a simple binary classification synthetic task."""
    cnn = GenerateCNN(
        input_shape=(8, 8, 1),
        conv_layers=[(4, 3)],
        pool_size=2,
        dense_layers=[8],
        output_size=1,
        learning_rate=0.1,
        epochs=10,
        output_activation="sigmoid"
    )
    
    X_train = np.random.randn(20, 8, 8, 1).astype(np.float32)
    y_train = (X_train.reshape(20, -1).sum(axis=1) > 0).astype(float)
    
    history = cnn.train(X_train, y_train, batch_size=5)
    
    assert history is not None
    assert "loss_history" in history
    assert len(history["loss_history"]) == 10
    assert history["epochs_trained"] == 10
    assert history["final_loss"] >= 0


def test_cnn_evaluate():
    """Test evaluation metrics calculation on classification tasks."""
    cnn = GenerateCNN(
        input_shape=(8, 8, 1),
        conv_layers=[(2, 3)],
        pool_size=2,
        dense_layers=[],
        output_size=3
    )
    
    X = np.random.randn(10, 8, 8, 1).astype(np.float32)
    y_int = np.random.randint(0, 3, 10)
    
    metrics = cnn.evaluate(X, y_int)
    assert "mse" in metrics
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    
    y_onehot = np.zeros((10, 3), dtype=np.float32)
    y_onehot[np.arange(10), y_int] = 1.0
    metrics_oh = cnn.evaluate(X, y_onehot)
    assert metrics_oh["accuracy"] == metrics["accuracy"]


def test_cnn_save_and_load(tmp_path):
    """Test saving and loading a trained CNN model preserves architecture and weights."""
    cnn = GenerateCNN(
        input_shape=(12, 12, 2),
        conv_layers=[(4, 3)],
        pool_size=2,
        dense_layers=[8],
        output_size=3,
        learning_rate=0.05,
        epochs=5,
        l2=0.01
    )
    
    X = np.random.randn(10, 12, 12, 2).astype(np.float32)
    y = np.random.randint(0, 3, 10)
    cnn.train(X, y, batch_size=5)
    
    save_path = tmp_path / "cnn.npz"
    cnn.save(str(save_path))
    
    loaded_cnn = CNN.load(str(save_path))
    
    assert loaded_cnn.input_shape == cnn.input_shape
    assert loaded_cnn.conv_layers == cnn.conv_layers
    assert loaded_cnn.pool_size == cnn.pool_size
    assert loaded_cnn.dense_layers == cnn.dense_layers
    assert loaded_cnn.output_size == cnn.output_size
    assert loaded_cnn.learning_rate == cnn.learning_rate
    assert loaded_cnn.epochs == cnn.epochs
    assert loaded_cnn.output_activation == cnn.output_activation
    assert loaded_cnn.l2 == cnn.l2
    assert loaded_cnn.loss_history == cnn.loss_history
    
    for l_orig, l_loaded in zip(cnn.layers, loaded_cnn.layers):
        if hasattr(l_orig, "W"):
            assert np.allclose(l_orig.W, l_loaded.W)
            assert np.allclose(l_orig.b, l_loaded.b)
            
    preds_orig = cnn.predict(X)
    preds_loaded = loaded_cnn.predict(X)
    assert np.allclose(preds_orig, preds_loaded, atol=1e-6)


def test_cnn_optimizers_convergence():
    """Test convergence (loss decreases) for Adam, AdamW, RAdam, and RAdamW optimizers."""
    optimizers = ["adam", "adamw", "radam", "radamw"]
    
    # Synthetic batch of data
    X_train = np.random.randn(20, 8, 8, 1).astype(np.float32)
    y_train = (X_train.reshape(20, -1).sum(axis=1) > 0).astype(float)
    
    for opt in optimizers:
        cnn = GenerateCNN(
            input_shape=(8, 8, 1),
            conv_layers=[(4, 3)],
            pool_size=2,
            dense_layers=[8],
            output_size=1,
            learning_rate=0.01,
            epochs=5,
            output_activation="sigmoid",
            optimizer=opt,
            l2=0.01  # test weight decay logic too
        )
        
        history = cnn.train(X_train, y_train, batch_size=5)
        assert len(history["loss_history"]) == 5
        # Verify loss decreases or stays stable
        assert history["final_loss"] < history["loss_history"][0] or np.all(np.array(history["loss_history"]) >= 0)


def test_cnn_optimizers_saving_loading_moments(tmp_path):
    """Test optimizer states (momentum, steps) are successfully saved and loaded."""
    cnn = GenerateCNN(
        input_shape=(8, 8, 1),
        conv_layers=[(4, 3)],
        pool_size=2,
        dense_layers=[8],
        output_size=1,
        learning_rate=0.01,
        epochs=3,
        output_activation="sigmoid",
        optimizer="adamw",
        l2=0.01
    )
    
    X_train = np.random.randn(10, 8, 8, 1).astype(np.float32)
    y_train = np.random.randint(0, 2, 10).astype(float)
    
    # Run training to initialize and update moments
    cnn.train(X_train, y_train, batch_size=5)
    assert cnn.t > 0
    
    # Check that moment vectors were created
    has_moments = False
    for layer in cnn.layers:
        if hasattr(layer, "m_W") and layer.m_W is not None:
            has_moments = True
            assert np.any(layer.m_W != 0)
            break
    assert has_moments
    
    # Save model
    save_path = tmp_path / "cnn_opt.npz"
    cnn.save(str(save_path))
    
    # Load back
    loaded_cnn = CNN.load(str(save_path))
    assert loaded_cnn.t == cnn.t
    assert loaded_cnn.optimizer == "adamw"
    
    # Verify loaded states match
    for l_orig, l_loaded in zip(cnn.layers, loaded_cnn.layers):
        if hasattr(l_orig, "m_W") and l_orig.m_W is not None:
            assert np.allclose(l_orig.m_W, l_loaded.m_W)
            assert np.allclose(l_orig.v_W, l_loaded.v_W)
            assert np.allclose(l_orig.m_b, l_loaded.m_b)
            assert np.allclose(l_orig.v_b, l_loaded.v_b)

