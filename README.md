# adAI (adaiml-tools)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Documentation Status](https://readthedocs.org/projects/adaiml-tools/badge/?version=latest)](https://adAI.readthedocs.io)

**adAI** (published as `adaiml-tools`) is a lightweight, modular, and high-performance machine learning library built from scratch in Python using NumPy, with an optional high-performance C++ backend. Designed to make AI/ML components accessible, transparent, and easy to understand, adAI lets you construct anything from simple single-layer Perceptrons to Deep Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs).

---

## 🚀 Key Features

- **Built from Scratch**: Zero deep learning framework dependencies—only raw Python and NumPy.
- **Deep MLP & CNN Support**: Full forward and backward propagation implementation for convolutional, dense, batch normalization, and dropout layers.
- **Advanced Optimizers**: Includes `SGD`, `Adam`, `AdamW`, `RAdam`, and `RAdamW`.
- **Learning Rate Schedulers**: Fine-tune training with `StepLR`, `ExponentialLR`, `CosineAnnealingLR`, and `ReduceLROnPlateau`.
- **A Rich Ecosystem of Components**:
  - **Activations**: Sigmoid, ReLU, Tanh, Linear, Softmax.
  - **Losses**: Mean Squared Error (MSE), Mean Absolute Error (MAE), Binary Cross Entropy, Categorical Cross Entropy, Hinge Loss.
  - **Layers**: Dense, Conv2D, Flatten, MaxPool2D, BatchNorm, Dropout.
- **C++ Backend**: Optional high-performance C++ backend with automatic differentiation, computational graph optimization, and CUDA support (see [cpp_backend/README.md](cpp_backend/README.md)).

---

## 📦 Installation

To install the latest stable version from PyPI:

```bash
pip install adaiml-tools
```

For development and building from source (Python only):

```bash
git clone https://github.com/artheidendureza/adAI.git
cd adAI
pip install -e ".[dev]"
```

To build with the optional C++ backend:

```bash
git clone https://github.com/artheidendureza/adAI.git
cd adAI
pip install -e ".[dev]"
# Build C++ extension (requires CMake and C++17 compiler)
pip install -e . --global-option=--cpp-backend
```

See [cpp_backend/README.md](cpp_backend/README.md) for detailed C++ backend build instructions.

---

## ⚡ Quick Start

adAI provides intuitive generators (`GeneratePerceptron`, `GenerateMLP`, `GenerateCNN`) to instantiate pre-configured networks instantly.

### 1. Single Perceptron
```python
import numpy as np
from adAI import GeneratePerceptron

# Instantiate a Perceptron for binary classification
perceptron = GeneratePerceptron(input_size=5, learning_rate=0.01, epochs=500)

# Generate synthetic train/val/test data
data = perceptron.generate_data(n_train=200, n_val=50, n_test=50)

# Train the model
history = perceptron.train(
    data['X_train'], data['y_train'],
    X_val=data['X_val'], y_val=data['y_val'],
    early_stopping=True, patience=10
)

# Evaluate and predict
metrics = perceptron.evaluate(data['X_test'], data['y_test'])
print(f"Test Accuracy: {metrics['accuracy']:.2%}")
```

### 2. Multi-Layer Perceptron (MLP)
```python
import numpy as np
from adAI import GenerateMLP, StepLR

# Define a 3-layer MLP for classification
mlp = GenerateMLP(
    input_size=10,
    hidden_layers=[64, 32],
    output_size=3,
    learning_rate=0.01,
    epochs=100,
    activation="relu",
    output_activation="softmax",
    dropout=0.2
)

# Set up learning rate scheduling
scheduler = StepLR(step_size=30, gamma=0.5)

# Custom training loop with scheduling
for epoch in range(100):
    mlp.learning_rate = scheduler.get_lr(mlp.learning_rate, epoch)
    history = mlp.train(X_train, y_train, batch_size=32, verbose=False)
```

### 3. Convolutional Neural Network (CNN)
```python
from adAI import GenerateCNN

# Build a CNN with two Conv layers and a Dense head
cnn = GenerateCNN(
    input_shape=(14, 14, 1), # Height, Width, Channels
    conv_layers=[(16, 3), (32, 3)], # (Filters, Kernel Size)
    dense_layers=[32],
    output_size=3,
    learning_rate=0.02,
    epochs=20,
    optimizer="adamw"
)

# Train the CNN
history = cnn.train(X_train, y_train, X_val=X_test, y_val=y_test, batch_size=16)

# Save/Load models easily
cnn.save("saved_cnn.npz")
```

---

## 🛠️ API & Configuration

### Supported Optimizers
Specify any of the following string aliases or class instances:
- `"sgd"` / `SGD`
- `"adam"` / `Adam`
- `"adamw"` / `AdamW`
- `"radam"` / `RAdam`
- `"radamw"` / `RAdamW`

### Supported Loss Functions
- `"mse"` / `mean_squared_error`
- `"mae"` / `mean_absolute_error`
- `"binary_crossentropy"` / `binary_cross_entropy`
- `"categorical_crossentropy"` / `categorical_cross_entropy`
- `"hinge"` / `hinge_loss`

### Learning Rate Schedulers
- `ConstantLR`
- `StepLR(step_size, gamma)`
- `ExponentialLR(gamma)`
- `CosineAnnealingLR(T_max, eta_min)`
- `ReduceLROnPlateau(factor, patience, min_lr)`

---

## 🧪 Testing

We use `pytest` for unit testing and verifying gradient computations. Run tests with coverage using:

```bash
# Run all tests
pytest

# Run tests with HTML coverage report
pytest --cov=src/adAI
```

To test the C++ backend integration:

```bash
# Run C++ integration tests
pytest test_cpp_integration.py

# Run C++ backend tests
cd cpp_backend/build
ctest
```

---

## 📁 Project Structure

```
adAI/
├── src/adAI/              # Main Python library source
│   ├── activations/       # Activation functions
│   ├── layers/            # Neural network layers
│   ├── losses/            # Loss functions
│   ├── optimizers/        # Optimization algorithms
│   └── schedulers/        # Learning rate schedulers
├── cpp_backend/           # High-performance C++ backend
│   ├── include/           # C++ header files
│   ├── src/               # C++ implementation
│   ├── tests/             # C++ unit tests
│   └── examples/          # C++ usage examples
├── tests/                 # Python unit tests
├── examples/              # Python usage examples
└── docs/                  # Documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide to learn how you can submit Pull Requests, report issues, or suggest new features.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.