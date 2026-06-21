# adaiml-tools, or adAI

A Python library for making AI/Machine Learning easier and more accessible. Start with simple neural network components like perceptrons and build from there. Title is adaiml-tools, but use adAI.

## Installation

```bash
pip install adAI
```

Or for development:

```bash
git clone https://github.com/yourusername/adAI.git
cd adAI
pip install -e ".[dev]"
```

## Quick Start - Generate a Perceptron

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

## Features

- **GeneratePerceptron()** - Automatically create and initialize a perceptron
- Multiple activation functions (sigmoid, relu, tanh)
- Easy training and prediction
- Built-in loss tracking
- Growing library of AI components

## Configuration Options

```python
perceptron = GeneratePerceptron(
    input_size=10,           # Number of input features
    learning_rate=0.01,      # Learning rate (default: 0.01)
    epochs=100,              # Number of training epochs (default: 100)
    activation="sigmoid"     # Activation: "sigmoid", "relu", "tanh" (default: "sigmoid")
)
```

## Documentation

For complete documentation, visit [adAI documentation](https://adAI.readthedocs.io).

## Testing

Run tests with pytest:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=src/adAI
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
