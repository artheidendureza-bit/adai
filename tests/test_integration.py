"""
Integration tests for adAI library
"""

import numpy as np
import pytest
import adAI


def test_perceptron_end_to_end():
    """Test complete perceptron workflow"""
    # Generate data
    perceptron = adAI.GeneratePerceptron(input_size=2, epochs=50)
    data = perceptron.generate_data(n_train=100, n_val=20, n_test=20)
    
    # Train
    history = perceptron.train(data['X_train'], data['y_train'], verbose=False)
    assert 'loss_history' in history
    assert len(history['loss_history']) > 0
    
    # Evaluate
    metrics = perceptron.evaluate(data['X_test'], data['y_test'])
    assert 'mse' in metrics
    assert metrics['mse'] >= 0
    
    # Save and load
    perceptron.save('test_perceptron.npz')
    loaded = adAI.Perceptron.load('test_perceptron.npz')
    assert loaded.input_size == perceptron.input_size


def test_mlp_end_to_end():
    """Test complete MLP workflow"""
    # Generate data
    X = np.random.randn(200, 5).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)
    
    # Split
    X_train, X_test, y_train, y_test = adAI.train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train
    mlp = adAI.GenerateMLP(input_size=5, hidden_layers=[8, 4], output_size=1, epochs=50)
    history = mlp.train(X_train, y_train, verbose=False)
    assert 'loss_history' in history
    
    # Evaluate
    metrics = mlp.evaluate(X_test, y_test)
    assert 'mse' in metrics
    
    # Save and load
    mlp.save('test_mlp.npz')
    loaded = adAI.MLP.load('test_mlp.npz')
    assert loaded.input_size == mlp.input_size


def test_cnn_end_to_end():
    """Test complete CNN workflow"""
    # Generate simple image data
    X = np.random.randn(100, 8, 8, 1).astype(np.float32)
    y = np.random.randint(0, 2, 100)
    
    # Split
    X_train, X_test, y_train, y_test = adAI.train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train
    cnn = adAI.GenerateCNN(
        input_shape=(8, 8, 1),
        conv_layers=[(4, 3)],
        dense_layers=[8],
        output_size=2,
        epochs=30
    )
    history = cnn.train(X_train, y_train, verbose=False)
    assert 'loss_history' in history
    
    # Evaluate
    metrics = cnn.evaluate(X_test, y_test)
    assert 'mse' in metrics
    
    # Save and load
    cnn.save('test_cnn.npz')
    loaded = adAI.CNN.load('test_cnn.npz')
    assert loaded.input_shape == cnn.input_shape


def test_data_preprocessing():
    """Test data preprocessing utilities"""
    X = np.random.randn(100, 5).astype(np.float32)
    
    # Normalize
    X_norm = adAI.normalize(X)
    assert X_norm.min() >= 0
    assert X_norm.max() <= 1
    
    # Standardize
    X_std = adAI.standardize(X)
    assert np.abs(X_std.mean()) < 0.1
    assert np.abs(X_std.std() - 1.0) < 0.1


def test_loss_functions():
    """Test loss functions"""
    y_true = np.array([1.0, 0.0, 1.0])
    y_pred = np.array([0.9, 0.1, 0.8])
    
    # MSE
    mse = adAI.mean_squared_error(y_true, y_pred)
    assert mse >= 0
    
    # Binary cross-entropy
    bce = adAI.binary_cross_entropy(y_true, y_pred)
    assert bce >= 0
    
    # Hinge loss
    y_true_hinge = np.array([1.0, -1.0, 1.0])
    hinge = adAI.hinge_loss(y_true_hinge, y_pred)
    assert hinge >= 0


def test_learning_rate_schedulers():
    """Test learning rate schedulers"""
    initial_lr = 0.1
    
    # StepLR
    scheduler = adAI.StepLR(step_size=10, gamma=0.5)
    lr = scheduler.get_lr(initial_lr, 5)
    assert lr == initial_lr
    lr = scheduler.get_lr(initial_lr, 10)
    assert lr == initial_lr * 0.5
    
    # ExponentialLR
    scheduler = adAI.ExponentialLR(gamma=0.9)
    lr = scheduler.get_lr(initial_lr, 10)
    assert lr < initial_lr
    
    # CosineAnnealingLR
    scheduler = adAI.CosineAnnealingLR(T_max=100)
    lr = scheduler.get_lr(initial_lr, 0)
    assert lr == initial_lr
    lr = scheduler.get_lr(initial_lr, 100)
    assert lr == scheduler.eta_min


def test_optimizers():
    """Test optimizer implementations"""
    params = np.array([1.0, 2.0, 3.0])
    grads = np.array([0.1, 0.2, 0.3])
    
    # SGD
    optimizer = adAI.SGD(learning_rate=0.01)
    updated = optimizer.update(params, grads)
    assert not np.array_equal(updated, params)
    
    # Adam
    optimizer = adAI.Adam(learning_rate=0.001)
    updated = optimizer.update(params, grads)
    assert not np.array_equal(updated, params)
    
    # AdamW
    optimizer = adAI.AdamW(learning_rate=0.001, weight_decay=0.01)
    updated = optimizer.update(params, grads)
    assert not np.array_equal(updated, params)


def test_batch_normalization():
    """Test batch normalization layer"""
    bn = adAI.BatchNormLayer(num_features=4)
    X = np.random.randn(10, 4).astype(np.float32)
    
    # Forward training
    out = bn.forward(X, training=True)
    assert out.shape == X.shape
    
    # Forward inference
    out = bn.forward(X, training=False)
    assert out.shape == X.shape


def test_dropout_layer():
    """Test dropout layer"""
    dropout = adAI.DropoutLayer(dropout_rate=0.5)
    X = np.random.randn(10, 4).astype(np.float32)
    
    # Training mode
    out = dropout.forward(X, training=True)
    assert out.shape == X.shape
    
    # Inference mode
    out = dropout.forward(X, training=False)
    assert np.array_equal(out, X)


def test_activation_functions():
    """Test activation functions"""
    x = np.array([-1.0, 0.0, 1.0])
    
    # Sigmoid
    out = adAI.sigmoid(x)
    assert out.min() > 0
    assert out.max() < 1
    
    # ReLU
    out = adAI.relu(x)
    assert out.min() >= 0
    
    # Tanh
    out = adAI.tanh_activation(x)
    assert out.min() >= -1
    assert out.max() <= 1
    
    # Softmax
    x_multi = np.array([[1.0, 2.0, 3.0]])
    out = adAI.softmax(x_multi)
    assert np.abs(out.sum() - 1.0) < 0.01


def test_get_activation():
    """Test get_activation function"""
    sigmoid_fn, sigmoid_deriv = adAI.get_activation('sigmoid')
    x = np.array([0.0])
    out = sigmoid_fn(x)
    assert out.shape == x.shape
    
    # Test invalid activation
    with pytest.raises(ValueError):
        adAI.get_activation('invalid')


def test_get_loss():
    """Test get_loss function"""
    loss_fn, loss_deriv = adAI.get_loss('mse')
    y_true = np.array([1.0])
    y_pred = np.array([0.5])
    loss = loss_fn(y_true, y_pred)
    assert loss >= 0
    
    # Test invalid loss
    with pytest.raises(ValueError):
        adAI.get_loss('invalid')


def test_get_optimizer():
    """Test get_optimizer function"""
    optimizer = adAI.get_optimizer('adam', learning_rate=0.001)
    assert isinstance(optimizer, adAI.Adam)
    
    # Test invalid optimizer
    with pytest.raises(ValueError):
        adAI.get_optimizer('invalid')
