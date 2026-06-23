try:
    from ._version import version as __version__
except ImportError:
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
from .snn import GenerateSNN, SNN
from .gnn import GenerateGNN, GNN, Graph
from .transformer import GenerateTransformer, Transformer
from .diffusion import GenerateDiffusionModel, DiffusionModel
from .rl import GeneratePPOAgent, PPOAgent, Environment
from .datasets import load_data, load_synthetic_cnn_data, load_synthetic_snn_data, load_synthetic_graph_data, load_synthetic_sequence_data
from .graph import Node, PlaceholderNode, ParameterNode, OpNode, Graph
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

# C++ Backend (optional)
try:
    from .cpp_backend import (
        is_cpp_backend_available,
        CppTensor,
        CppGraph,
        create_graph,
        tensor_operations,
    )
    _cpp_backend_available = True
except ImportError:
    _cpp_backend_available = False

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
    'GenerateSNN',
    'SNN',
    'GenerateGNN',
    'GNN',
    'Graph',
    'GenerateTransformer',
    'Transformer',
    'GenerateDiffusionModel',
    'DiffusionModel',
    'GeneratePPOAgent',
    'PPOAgent',
    'Environment',
    'load_data',
    'load_synthetic_cnn_data',
    'load_synthetic_snn_data',
    'load_synthetic_graph_data',
    'load_synthetic_sequence_data',
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
    'Node',
    'PlaceholderNode',
    'ParameterNode',
    'OpNode',
    'Graph',
]

# Add C++ backend exports if available
if _cpp_backend_available:
    __all__.extend([
        'is_cpp_backend_available',
        'CppTensor',
        'CppGraph',
        'create_graph',
        'tensor_operations',
    ])
