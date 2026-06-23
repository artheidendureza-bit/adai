"""
Python wrapper for adAI C++ backend
Provides a Pythonic interface to the C++ computational graph implementation
"""

import numpy as np
from typing import Optional, Dict, List, Union, Callable, Any
import warnings

# Try to import the C++ extension
try:
    import adaicpp_py as _cpp
    _CPP_BACKEND_AVAILABLE = True
except ImportError:
    _CPP_BACKEND_AVAILABLE = False
    warnings.warn(
        "adAI C++ backend not available. Falling back to Python implementation. "
        "To build the C++ backend, run: python cpp_backend/build_ext.py"
    )


def is_cpp_backend_available() -> bool:
    """Check if the C++ backend is available"""
    return _CPP_BACKEND_AVAILABLE


class CppTensor:
    """Wrapper for C++ Tensor with numpy interoperability"""
    
    def __init__(self, cpp_tensor):
        if not _CPP_BACKEND_AVAILABLE:
            raise RuntimeError("C++ backend is not available")
        self._tensor = cpp_tensor
    
    @classmethod
    def from_numpy(cls, array: np.ndarray, device=None):
        """Create CppTensor from numpy array"""
        if not _CPP_BACKEND_AVAILABLE:
            raise RuntimeError("C++ backend is not available")
        
        device_manager = _cpp.DeviceManager.instance()
        if device is None:
            device = device_manager.get_cpu_device()
        
        cpp_tensor = _cpp.numpy_to_tensor(array, device)
        return cls(cpp_tensor)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        if not _CPP_BACKEND_AVAILABLE:
            raise RuntimeError("C++ backend is not available")
        return _cpp.tensor_to_numpy(self._tensor)
    
    @property
    def shape(self):
        return self._tensor.shape()
    
    @property
    def ndim(self):
        return self._tensor.ndim()
    
    @property
    def size(self):
        return self._tensor.size()
    
    def __repr__(self):
        return f"CppTensor(shape={self.shape}, dtype=float32)"


class CppGraph:
    """Wrapper for C++ Graph with Pythonic interface"""
    
    def __init__(self):
        if not _CPP_BACKEND_AVAILABLE:
            raise RuntimeError("C++ backend is not available")
        
        # Initialize the C++ backend
        _cpp.initialize()
        _cpp.register_standard_operations()
        
        self._graph = _cpp.Graph()
        self._device_manager = _cpp.DeviceManager.instance()
        self._cpu_device = self._device_manager.get_cpu_device()
    
    def add_placeholder(self, name: str, device: str = "CPU") -> Any:
        """Add a placeholder node to the graph"""
        device_obj = self._cpu_device if device == "CPU" else None
        return self._graph.add_placeholder(name, device_obj)
    
    def add_parameter(self, name: str, value: np.ndarray, device: str = "CPU") -> Any:
        """Add a parameter node to the graph"""
        device_obj = self._cpu_device if device == "CPU" else None
        cpp_tensor = _cpp.numpy_to_tensor(value, device_obj)
        cpp_tensor_shared = _cpp.Tensor(value.shape, value, device_obj)
        return self._graph.add_parameter(name, cpp_tensor_shared)
    
    def add_op(
        self,
        name: str,
        op: Union[str, Callable],
        inputs: List[Any],
        is_static: Optional[bool] = None,
        device: str = "CPU"
    ) -> Any:
        """Add an operation node to the graph"""
        device_obj = self._cpu_device if device == "CPU" else None
        
        if isinstance(op, str):
            # Use registered operation
            forward_fn = _cpp.OpRegistry.instance().get_forward_fn(op)
            backward_fn = _cpp.OpRegistry.instance().get_backward_fn(op)
        else:
            # Use custom callable (not fully supported yet)
            forward_fn = lambda inputs: op(*[i.value() for i in inputs])
            backward_fn = None
        
        return self._graph.add_op(name, inputs, forward_fn, backward_fn, is_static, device_obj)
    
    def compile(self):
        """Compile the graph"""
        self._graph.compile()
    
    def forward(self, feeds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run forward pass with input feeds"""
        # Convert numpy feeds to C++ tensors
        cpp_feeds = {}
        for name, value in feeds.items():
            cpp_tensor = _cpp.numpy_to_tensor(value, self._cpu_device)
            cpp_feeds[name] = cpp_tensor
        
        self._graph.forward(cpp_feeds)
        
        # Get results and convert back to numpy
        results = {}
        for name, node in self._graph.get_node(name):
            if node.value():
                results[name] = _cpp.tensor_to_numpy(node.value())
        
        return results
    
    def backward(self, target_node_name: str):
        """Run backward pass for gradient computation"""
        self._graph.backward(target_node_name)
    
    def optimize(self, outputs: List[str]):
        """Optimize the graph (dead code elimination, constant folding)"""
        self._graph.optimize(outputs)
    
    def get_value(self, name: str) -> Optional[np.ndarray]:
        """Get the value of a node"""
        node = self._graph.get_node(name)
        if node.value():
            return _cpp.tensor_to_numpy(node.value())
        return None
    
    def get_grad(self, name: str) -> Optional[np.ndarray]:
        """Get the gradient of a node"""
        node = self._graph.get_node(name)
        if node.grad():
            return _cpp.tensor_to_numpy(node.grad())
        return None
    
    def zero_grad(self):
        """Zero all gradients"""
        self._graph.zero_grad()
    
    def to_mermaid(self) -> str:
        """Generate Mermaid diagram"""
        return self._graph.to_mermaid()
    
    def print_layout(self):
        """Print graph layout"""
        self._graph.print_layout()
    
    def save(self, path: str):
        """Save graph to file"""
        self._graph.save(path)
    
    @classmethod
    def load(cls, path: str) -> "CppGraph":
        """Load graph from file"""
        instance = cls()
        instance._graph = _cpp.Graph.load(path)
        return instance


# Convenience functions
def create_graph() -> CppGraph:
    """Create a new C++ graph"""
    return CppGraph()


def tensor_operations():
    """Return available tensor operations from C++ backend"""
    if not _CPP_BACKEND_AVAILABLE:
        return {}
    
    return {
        'add': _cpp.add,
        'sub': _cpp.sub,
        'mul': _cpp.mul,
        'div': _cpp.div,
        'relu': _cpp.relu,
        'sigmoid': _cpp.sigmoid,
        'tanh': _cpp.tanh,
        'matmul': _cpp.matmul,
        'sum': _cpp.sum,
        'mean': _cpp.mean,
    }


# Export main classes and functions
__all__ = [
    'is_cpp_backend_available',
    'CppTensor',
    'CppGraph',
    'create_graph',
    'tensor_operations',
]
