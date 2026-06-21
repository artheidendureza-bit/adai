# adAI Documentation

Welcome to the adAI library documentation!

## Overview

adAI is a Python library for making AI/Machine Learning easier and more accessible. Start with simple neural network components like perceptrons and build from there. The library provides pure NumPy implementations of:

- **Perceptron**: Single-layer neural network for binary classification
- **MLP (Multi-Layer Perceptron)**: Fully connected neural networks with hidden layers
- **CNN (Convolutional Neural Network)**: Deep learning models for image data

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](../CONTRIBUTING.md)

## Installation

```bash
pip install adai
```

Or for development:

```bash
git clone https://github.com/artheidendureza/adAI.git
cd adAI
pip install -e ".[dev]"
```

## Quick Start

### Perceptron Example

```python
import numpy as np
from adAI import GeneratePerceptron

# Create a perceptron with 5 input features
perceptron = GeneratePerceptron(input_size=5, epochs=100)

# Train on your data
X_train = np.random.randn(100, 5)
y_train = np.random.randint(0, 2, 100)
perceptron.train(X_train, y_train)

# Make predictions
predictions = perceptron.predict(X_test)
```

### MLP Example

```python
from adAI import GenerateMLP

# Create an MLP with hidden layers
mlp = GenerateMLP(
    input_size=10,
    hidden_layers=[32, 16],
    output_size=1,
    learning_rate=0.01,
    epochs=100
)

# Train the model
history = mlp.train(X_train, y_train)
```

### CNN Example

```python
from adAI import GenerateCNN

# Create a CNN for image classification
cnn = GenerateCNN(
    input_shape=(28, 28, 1),
    conv_layers=[(32, 3), (64, 3)],
    dense_layers=[128],
    output_size=10,
    learning_rate=0.001,
    epochs=50
)

# Train the model
history = cnn.train(X_train, y_train)
```

## API Reference

### Core Classes

- [`BaseModel`](#basemodel) - Base class for all models
- [`Perceptron`](#perceptron) - Single-layer neural network
- [`MLP`](#mlp) - Multi-layer perceptron
- [`CNN`](#cnn) - Convolutional neural network

### Activation Functions

- [`sigmoid`](#sigmoid) - Sigmoid activation
- [`relu`](#relu) - ReLU activation
- [`tanh_activation`](#tanh_activation) - Tanh activation
- [`linear`](#linear) - Linear (identity) activation
- [`softmax`](#softmax) - Softmax activation
- [`get_activation`](#get_activation) - Get activation function by name

### BaseModel

Base class providing common functionality for all models.

**Methods:**
- `summary()` - Print model summary
- `train(X, y, X_val=None, y_val=None, ...)` - Train the model
- `predict(X)` - Make predictions
- `evaluate(X, y)` - Evaluate on test data
- `save(path)` - Save model to file
- `load(path)` - Load model from file (classmethod)

### Perceptron

Single-layer neural network for binary classification.

**Parameters:**
- `input_size` (int): Number of input features
- `learning_rate` (float): Learning rate (default: 0.01)
- `epochs` (int): Number of training epochs (default: 100)
- `activation` (str): Activation function - "sigmoid", "relu", "tanh" (default: "sigmoid")

**Methods:**
- All BaseModel methods
- `generate_data(n_train, n_val, n_test, seed)` - Generate synthetic data

### MLP

Multi-layer perceptron with configurable hidden layers.

**Parameters:**
- `input_size` (int): Number of input features
- `hidden_layers` (list): List of hidden layer sizes
- `output_size` (int): Number of output units
- `learning_rate` (float): Learning rate (default: 0.01)
- `epochs` (int): Number of training epochs (default: 100)
- `activation` (str): Hidden layer activation (default: "relu")
- `output_activation` (str): Output activation - "sigmoid", "relu", "linear", "softmax"
- `l2` (float): L2 regularization coefficient (default: 0.0)
- `dropout` (float): Dropout rate (default: 0.0)

### CNN

Convolutional neural network for image data.

**Parameters:**
- `input_shape` (tuple): Shape of input images (height, width, channels)
- `conv_layers` (list): List of (filters, kernel_size) tuples
- `dense_layers` (list): List of dense layer sizes
- `output_size` (int): Number of output units
- `pool_size` (int): Pooling size (default: 2)
- `learning_rate` (float): Learning rate (default: 0.01)
- `epochs` (int): Number of training epochs (default: 100)
- `output_activation` (str): Output activation function
- `l2` (float): L2 regularization coefficient (default: 0.0)
- `optimizer` (str): Optimizer - "sgd", "adam", "adamw", "radam", "radamw" (default: "adam")

## Features

- **Pure NumPy Implementation**: No deep learning framework dependencies
- **Multiple Activation Functions**: Sigmoid, ReLU, Tanh, Linear, Softmax
- **Flexible Architecture**: Configurable layer sizes and hyperparameters
- **Training Features**: Mini-batch training, early stopping, validation
- **Regularization**: L2 regularization and dropout support
- **Multiple Optimizers**: SGD, Adam, AdamW, RAdam, RAdamW
- **Model Persistence**: Save and load trained models
- **Evaluation Metrics**: MSE, MAE, RMSE, Accuracy

## License

MIT License - See LICENSE file for details
