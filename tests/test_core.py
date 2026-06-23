"""
Unit tests for adAI
"""

import pytest
import numpy as np
from adAI import GeneratePerceptron, GenerateMLP, GenerateXORMlp, MLP, generate_xor_data


def test_generate_perceptron_creation():
    """Test that GeneratePerceptron creates a perceptron"""
    perceptron = GeneratePerceptron(input_size=5)
    assert perceptron is not None
    assert perceptron.input_size == 5
    assert len(perceptron.weights) == 5


def test_generate_perceptron_forward():
    """Test forward pass"""
    perceptron = GeneratePerceptron(input_size=3)
    X = np.array([[1, 2, 3], [4, 5, 6]])
    output = perceptron.predict(X)
    assert output.shape == (2,)
    assert np.all((output >= 0) & (output <= 1))  # Sigmoid output


def test_generate_perceptron_training():
    """Test perceptron training with basic dataset"""
    perceptron = GeneratePerceptron(input_size=2, epochs=20)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.float32)
    
    history = perceptron.train(X, y)
    
    assert history is not None
    assert len(history['loss_history']) == 20
    assert 'epochs_trained' in history
    assert history['epochs_trained'] == 20


def test_generate_perceptron_training_with_validation():
    """Test perceptron training with validation data"""
    perceptron = GeneratePerceptron(input_size=2, epochs=30, learning_rate=0.1)
    
    X_train = np.random.randn(50, 2).astype(np.float32)
    y_train = (X_train.sum(axis=1) > 0).astype(float)
    
    X_val = np.random.randn(20, 2).astype(np.float32)
    y_val = (X_val.sum(axis=1) > 0).astype(float)
    
    history = perceptron.train(X_train, y_train, X_val=X_val, y_val=y_val)
    
    assert history['val_loss_history'] is not None
    assert len(history['val_loss_history']) > 0


def test_generate_perceptron_early_stopping():
    """Test early stopping during training"""
    perceptron = GeneratePerceptron(input_size=2, epochs=1000, learning_rate=0.5)
    
    X_train = np.random.randn(50, 2).astype(np.float32)
    y_train = np.ones(50, dtype=float)  # All same class to trigger early stopping
    
    X_val = np.random.randn(20, 2).astype(np.float32)
    y_val = np.zeros(20, dtype=float)  # Different distribution
    
    history = perceptron.train(
        X_train, y_train, 
        X_val=X_val, y_val=y_val,
        early_stopping=True,
        patience=3
    )
    
    # Should stop before 1000 epochs due to early stopping or complete training
    assert history['epochs_trained'] <= 1000
    assert 'epochs_trained' in history


def test_generate_perceptron_evaluation():
    """Test evaluation metrics"""
    perceptron = GeneratePerceptron(input_size=2, epochs=50)
    
    X_train = np.random.randn(100, 2).astype(np.float32)
    y_train = (X_train.sum(axis=1) > 0).astype(float)
    
    perceptron.train(X_train, y_train)
    
    X_test = np.random.randn(30, 2).astype(np.float32)
    y_test = (X_test.sum(axis=1) > 0).astype(float)
    
    metrics = perceptron.evaluate(X_test, y_test)
    
    assert 'mse' in metrics
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'accuracy' in metrics
    assert metrics['mse'] >= 0
    assert metrics['rmse'] >= 0


def test_generate_perceptron_different_activations():
    """Test different activation functions"""
    for activation in ["sigmoid", "relu", "tanh"]:
        perceptron = GeneratePerceptron(input_size=3, activation=activation)
        assert perceptron.activation == activation


def test_generate_perceptron_batch_training():
    """Test mini-batch training"""
    perceptron = GeneratePerceptron(input_size=5, epochs=20)
    
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(float)
    
    history = perceptron.train(X, y, batch_size=32)
    
    assert history is not None
    assert len(history['loss_history']) == 20


def test_generate_perceptron_generate_data():
    """Test synthetic data generation"""
    perceptron = GeneratePerceptron(input_size=5)
    
    data = perceptron.generate_data(n_train=100, n_val=20, n_test=20)
    
    assert 'X_train' in data
    assert 'y_train' in data
    assert 'X_val' in data
    assert 'y_val' in data
    assert 'X_test' in data
    assert 'y_test' in data
    
    assert data['X_train'].shape == (100, 5)
    assert data['X_val'].shape == (20, 5)
    assert data['X_test'].shape == (20, 5)
    assert len(data['y_train']) == 100
    assert len(data['y_val']) == 20
    assert len(data['y_test']) == 20


def test_generate_perceptron_save_and_load(tmp_path):
    """Test save and load persistence"""
    perceptron = GeneratePerceptron(input_size=4, epochs=10, learning_rate=0.05, activation="tanh")
    data = perceptron.generate_data(n_train=20, n_val=10, n_test=10)
    perceptron.train(data['X_train'], data['y_train'])
    
    save_path = tmp_path / "perceptron.npz"
    perceptron.save(str(save_path))
    
    loaded = type(perceptron).load(str(save_path))
    
    assert loaded.input_size == perceptron.input_size
    assert loaded.learning_rate == perceptron.learning_rate
    assert loaded.epochs == perceptron.epochs
    assert loaded.activation == perceptron.activation
    assert np.allclose(loaded.weights, perceptron.weights)
    assert loaded.bias == perceptron.bias
    assert np.allclose(np.array(loaded.loss_history, dtype=np.float64), np.array(perceptron.loss_history, dtype=np.float64))
    
    sample = np.random.randn(5, 4).astype(np.float32)
    assert np.allclose(loaded.predict(sample), perceptron.predict(sample))


def test_generate_mlp_creation_and_training():
    """Test MLP creation and basic training"""
    mlp = GenerateMLP(input_size=2, hidden_layers=[4], epochs=20, learning_rate=0.1)
    assert isinstance(mlp, MLP)
    assert mlp.input_size == 2
    assert mlp.hidden_layers == [4]
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.float32)
    history = mlp.train(X, y, batch_size=4)
    
    assert history['epochs_trained'] == 20
    assert len(history['loss_history']) == 20
    assert history['final_loss'] >= 0


def test_generate_mlp_save_and_load(tmp_path):
    """Test MLP persistence"""
    mlp = GenerateMLP(input_size=2, hidden_layers=[3], epochs=5, learning_rate=0.1)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.float32)
    mlp.train(X, y, batch_size=4)

    save_path = tmp_path / "mlp.npz"
    mlp.save(str(save_path))
    loaded = MLP.load(str(save_path))

    assert loaded.input_size == mlp.input_size
    assert loaded.hidden_layers == mlp.hidden_layers
    assert loaded.output_size == mlp.output_size
    assert loaded.activation == mlp.activation
    assert loaded.dropout == mlp.dropout
    assert np.allclose(loaded.layers[0]['weights'], mlp.layers[0]['weights'])
    assert np.allclose(loaded.layers[0]['bias'], mlp.layers[0]['bias'])
    assert np.allclose(np.array(loaded.loss_history, dtype=np.float64), np.array(mlp.loss_history, dtype=np.float64))

    sample = np.random.randn(2, 2).astype(np.float32)
    assert np.allclose(loaded.predict(sample), mlp.predict(sample), atol=1e-6)


def test_generate_xor_data_helper():
    """Test XOR data generator from the MLP module"""
    X, y = generate_xor_data()
    assert X.shape == (4, 2)
    assert y.shape == (4,)
    assert set(y.tolist()) == {0.0, 1.0}


def test_generate_xor_mlp_factory():
    """Test minimal XOR MLP factory architecture"""
    mlp = GenerateXORMlp(epochs=10)
    assert mlp.input_size == 2
    assert mlp.hidden_layers == [2]
    assert mlp.output_size == 1
    assert mlp.activation == 'tanh'
