"""
adAI - A Python library for making AI easier
"""

__version__ = "0.1.0"
__author__ = "Artheiden"
__email__ = "artheidendureza@gmail.com"

from .perceptron import (
    GeneratePerceptron,
    Perceptron,
    generate_logic_gate_data,
    generate_not_data,
    generate_or_data,
    generate_nor_data,
    generate_and_data,
    generate_nand_data,
)
from .mlp import GenerateMLP, GenerateXORMlp, MLP, generate_xor_data
from .cnn import GenerateCNN, CNN
from .base import BaseModel
from .activations import (
    sigmoid,
    relu,
    tanh_activation,
    linear,
    softmax,
    get_activation,
)
from .core import train_test_split, normalize, standardize
from .losses import (
    mean_squared_error,
    mean_absolute_error,
    binary_cross_entropy,
    categorical_cross_entropy,
    hinge_loss,
    get_loss,
)
from .schedulers import (
    LRScheduler,
    ConstantLR,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)
from .layers import BatchNormLayer, DropoutLayer
from .optimizers import (
    Optimizer,
    SGD,
    Adam,
    AdamW,
    RAdam,
    RAdamW,
    get_optimizer,
)

__all__ = [
    'GeneratePerceptron',
    'Perceptron',
    'generate_logic_gate_data',
    'generate_not_data',
    'generate_or_data',
    'generate_nor_data',
    'generate_and_data',
    'generate_nand_data',
    'GenerateMLP',
    'GenerateXORMlp',
    'MLP',
    'generate_xor_data',
    'GenerateCNN',
    'CNN',
    'BaseModel',
    'sigmoid',
    'relu',
    'tanh_activation',
    'linear',
    'softmax',
    'get_activation',
    'train_test_split',
    'normalize',
    'standardize',
    'mean_squared_error',
    'mean_absolute_error',
    'binary_cross_entropy',
    'categorical_cross_entropy',
    'hinge_loss',
    'get_loss',
    'LRScheduler',
    'ConstantLR',
    'StepLR',
    'ExponentialLR',
    'CosineAnnealingLR',
    'ReduceLROnPlateau',
    'BatchNormLayer',
    'DropoutLayer',
    'Optimizer',
    'SGD',
    'Adam',
    'AdamW',
    'RAdam',
    'RAdamW',
    'get_optimizer',
]
