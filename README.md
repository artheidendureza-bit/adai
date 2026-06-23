# adAI (adaiml-tools)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Documentation Status](https://readthedocs.org/projects/adaiml-tools/badge/?version=latest)](https://adAI.readthedocs.io)

**adAI** (published as `adaiml-tools`) is a lightweight, modular, and high-performance machine learning library built from scratch in Python using NumPy, with an optional high-performance C++ backend. Designed to make AI/ML components accessible, transparent, and easy to understand, adAI lets you construct anything from simple single-layer Perceptrons to Deep Multi-Layer Perceptrons (MLPs), Convolutional Neural Networks (CNNs), Spiking Neural Networks (SNNs), Graph Neural Networks (GNNs), Transformers, Diffusion Models, and Reinforcement Learning agents.

---

## 🚀 Key Features

- **Built from Scratch**: Zero deep learning framework dependencies—only raw Python and NumPy.
- **Deep MLP & CNN Support**: Full forward and backward propagation implementation for convolutional, dense, batch normalization, and dropout layers.
- **Spiking Neural Networks (SNN)**: Leaky Integrate-and-Fire (LIF) neurons with rate and Poisson encoding, time-stepped simulation, and configurable reset modes.
- **Graph Neural Networks (GNN)**: Graph Convolutional Networks (GCN) for node classification with message passing on graph structures.
- **Transformers**: Multi-head attention, positional encoding, and feed-forward networks for sequence processing tasks.
- **Diffusion Models**: Denoising Diffusion Probabilistic Models (DDPM) for generative tasks with forward/reverse diffusion processes.
- **Reinforcement Learning**: PPO (Proximal Policy Optimization) agents with policy and value networks for environment interaction.
- **Advanced Optimizers**: Includes `SGD`, `Adam`, `AdamW`, `RAdam`, and `RAdamW`.
- **Learning Rate Schedulers**: Fine-tune training with `StepLR`, `ExponentialLR`, `CosineAnnealingLR`, and `ReduceLROnPlateau`.
- **A Rich Ecosystem of Components**:
  - **Activations**: Sigmoid, ReLU, Tanh, Linear, Softmax.
  - **Losses**: Mean Squared Error (MSE), Mean Absolute Error (MAE), Binary Cross Entropy, Categorical Cross Entropy, Hinge Loss.
  - **Layers**: Dense, Conv2D, Flatten, MaxPool2D, BatchNorm, Dropout, Spiking Layers, Graph Convolutional Layers.
- **C++ Backend**: Optional high-performance C++ backend with automatic differentiation, computational graph optimization, and CUDA support (see [cpp_backend/README.md](cpp_backend/README.md)).

---

## 📦 Installation

To install the latest stable version from PyPI:

```bash
pip install adaiml-tools
```

## 📊 Datasets

adAI includes built-in synthetic approximations of popular ML datasets for easy experimentation:

**Classification:**
- `iris` - Iris flower classification (150 samples, 4 features, 3 classes)
- `wine` - Wine classification (178 samples, 13 features, 3 classes)
- `breast_cancer` - Breast cancer classification (569 samples, 30 features, 2 classes)
- `adult` - Adult/Census Income (48842 samples, 14 features, 2 classes)
- `heart_disease` - Heart Disease (303 samples, 13 features, 2 classes)
- `bank_marketing` - Bank Marketing (45211 samples, 16 features, 2 classes)

**Regression:**
- `boston_housing` - Boston housing regression (506 samples, 13 features)
- `diabetes` - Diabetes regression (442 samples, 10 features)
- `california_housing` - California housing regression (20640 samples, 8 features)
- `energy` - Energy efficiency regression (768 samples, 8 features)

**Image (synthetic):**
- `mnist` - MNIST digits (70000 samples, 784 features, 10 classes)
- `fashion_mnist` - Fashion MNIST (70000 samples, 784 features, 10 classes)
- `cifar10` - CIFAR-10 (60000 samples, 3072 features, 10 classes)
- `cifar100` - CIFAR-100 (60000 samples, 3072 features, 100 classes)
- `imagenet` - ImageNet (10000 samples, 150528 features, 1000 classes)

**Text (synthetic):**
- `imdb` - IMDB Reviews (50000 samples, 5000 features, 2 classes)
- `penn_treebank` - Penn Treebank (42068 samples, 10000 features)

**Spiking (synthetic):**
- `ssc` - Spiking Speech Commands (105829 samples, 700 features, 35 classes)
- `shd` - Spiking Heidelberg Digits (20282 samples, 256 features, 20 classes)

**Synthetic:**
- `synthetic` - Generate synthetic classification data
- `synthetic_reg` - Generate synthetic regression data

**Important Note:** All built-in datasets are synthetic approximations for educational purposes. They mimic the structure and statistics of real datasets but are not the actual data. For production use or research, please download the original datasets from their official sources.

**Usage:**
```python
from adAI import load_data, GenerateMLP

# Load built-in dataset
X_train, X_test, y_train, y_test, scaler = load_data('iris')

# Generate synthetic data
X_train, X_test, y_train, y_test, scaler = load_data('synthetic', n_samples=100, n_features=4, n_classes=3)

# Use custom data
X_train, X_test, y_train, y_test, scaler = load_data(X=X, y=y)
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

### 4. Spiking Neural Network (SNN)
```python
from adAI import GenerateSNN

# Build an SNN with LIF neurons
snn = GenerateSNN(
    input_size=10,
    hidden_layers=[32, 16],
    output_size=1,
    time_steps=20,
    threshold=1.0,
    leak=0.1,
    reset_mode="subtract",
    encoding="rate"
)

# Generate synthetic data
data = snn.generate_data(n_train=500, n_val=100, n_test=100)

# Train the SNN
history = snn.train(
    data['X_train'], data['y_train'],
    X_val=data['X_val'], y_val=data['y_val'],
    early_stopping=True, patience=10
)

# Evaluate
metrics = snn.evaluate(data['X_test'], data['y_test'])
print(f"Test Accuracy: {metrics['accuracy']:.2%}")
```

### 5. Graph Neural Network (GNN)
```python
from adAI import GenerateGNN, Graph

# Create a graph with node features and adjacency matrix
node_features = np.random.randn(10, 16).astype(np.float32)
adjacency_matrix = np.random.randint(0, 2, (10, 10)).astype(np.float32)
graph = Graph(node_features, adjacency_matrix)

# Build a GNN
gnn = GenerateGNN(
    input_dim=16,
    hidden_dims=[32, 16],
    output_dim=4,
    activation="relu"
)

# Train the GNN
labels = np.random.randint(0, 4, 10)
mask = np.array([True] * 5 + [False] * 5)  # Only first 5 nodes labeled
history = gnn.train(graph, labels, mask=mask)

# Predict
predictions = gnn.predict(graph)
```

### 6. Transformer
```python
from adAI import GenerateTransformer

# Build a Transformer
transformer = GenerateTransformer(
    vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_seq_len=128
)

# Train with sequences
X_train = np.random.randint(0, 10000, (100, 32)).astype(np.int32)
y_train = np.random.randint(0, 10000, (100, 32)).astype(np.int32)
history = transformer.train(X_train, y_train, batch_size=16)

# Predict
predictions = transformer.predict(X_train)
```

### 7. Diffusion Model
```python
from adAI import GenerateDiffusionModel

# Build a Diffusion Model
diffusion = GenerateDiffusionModel(
    data_dim=64,
    hidden_dims=[128, 256],
    n_timesteps=1000
)

# Train
X_train = np.random.randn(500, 64).astype(np.float32)
history = diffusion.train(X_train, batch_size=32)

# Generate samples
samples = diffusion.sample(n_samples=100)
```

### 8. Reinforcement Learning (PPO Agent)
```python
from adAI import GeneratePPOAgent, Environment

# Create environment
env = Environment(state_dim=4, action_dim=2, max_steps=100)

# Build PPO Agent
agent = GeneratePPOAgent(
    state_dim=4,
    action_dim=2,
    hidden_dims=[64, 32],
    gamma=0.99
)

# Train
history = agent.train(env, n_episodes=1000, verbose=True)

# Evaluate
metrics = agent.evaluate(env, n_episodes=10)
print(f"Average Reward: {metrics['average_reward']:.2f}")
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