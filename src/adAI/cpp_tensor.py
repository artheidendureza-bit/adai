"""
C++ Tensor wrapper for adAI.
Provides a Python interface to the C++ Tensor implementation.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from .backend import get_backend, Backend, get_cpp_module, is_cpp_available


class CppTensor:
    """
    Wrapper for C++ Tensor class.
    Provides a Python-friendly interface to the high-performance C++ tensor operations.
    """
    
    def __init__(self, data: Union[np.ndarray, List[float], Tuple[int, ...]], 
                 device: Optional[str] = None):
        """
        Initialize a C++ tensor.
        
        Args:
            data: Either a numpy array, list of floats, or shape tuple
            device: Device to use ('cpu' or 'cuda')
        """
        if not is_cpp_available():
            raise RuntimeError("C++ backend is not available")
        
        cpp = get_cpp_module()
        
        # Handle different input types
        if isinstance(data, np.ndarray):
            # Convert numpy array to C++ tensor
            self._device = cpp.cpu_device() if device != 'cuda' else None
            self._tensor = cpp.from_numpy(data, self._device)
        elif isinstance(data, (list, tuple)):
            if all(isinstance(x, (int, float)) for x in data):
                # List of floats - create 1D tensor
                arr = np.array(data, dtype=np.float32)
                self._device = cpp.cpu_device() if device != 'cuda' else None
                self._tensor = cpp.from_numpy(arr, self._device)
            else:
                # Shape tuple - create empty tensor
                shape = list(data)
                self._device = cpp.cpu_device() if device != 'cuda' else None
                self._tensor = cpp.Tensor(shape, self._device)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        return tuple(self._tensor.shape)
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return self._tensor.ndim
    
    @property
    def size(self) -> int:
        """Get total number of elements."""
        return self._tensor.size
    
    @property
    def device(self) -> str:
        """Get device as string."""
        return "cpu" if self._tensor.device.type.name == "CPU" else "cuda"
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self._tensor.numpy())
    
    def fill(self, value: float) -> 'CppTensor':
        """Fill tensor with value."""
        self._tensor.fill(value)
        return self
    
    def zero(self) -> 'CppTensor':
        """Fill tensor with zeros."""
        self._tensor.zero()
        return self
    
    def copy_from(self, other: 'CppTensor') -> 'CppTensor':
        """Copy data from another tensor."""
        self._tensor.copy_from(other._tensor)
        return self
    
    def reshape(self, new_shape: Tuple[int, ...]) -> 'CppTensor':
        """Reshape tensor."""
        result = self._tensor.reshape(list(new_shape))
        return CppTensor._from_cpp_tensor(result)
    
    def transpose(self, dims: Tuple[int, ...]) -> 'CppTensor':
        """Transpose tensor."""
        result = self._tensor.transpose(list(dims))
        return CppTensor._from_cpp_tensor(result)
    
    @staticmethod
    def _from_cpp_tensor(cpp_tensor) -> 'CppTensor':
        """Create CppTensor from C++ tensor object."""
        wrapper = CppTensor.__new__(CppTensor)
        wrapper._tensor = cpp_tensor
        wrapper._device = cpp_tensor.device
        return wrapper
    
    @staticmethod
    def zeros(shape: Tuple[int, ...], device: str = 'cpu') -> 'CppTensor':
        """Create tensor filled with zeros."""
        cpp = get_cpp_module()
        dev = cpp.cpu_device() if device == 'cpu' else None
        tensor = cpp.Tensor.zeros(list(shape), dev)
        return CppTensor._from_cpp_tensor(tensor)
    
    @staticmethod
    def ones(shape: Tuple[int, ...], device: str = 'cpu') -> 'CppTensor':
        """Create tensor filled with ones."""
        cpp = get_cpp_module()
        dev = cpp.cpu_device() if device == 'cpu' else None
        tensor = cpp.Tensor.ones(list(shape), dev)
        return CppTensor._from_cpp_tensor(tensor)
    
    @staticmethod
    def randn(shape: Tuple[int, ...], device: str = 'cpu') -> 'CppTensor':
        """Create tensor with random normal values."""
        cpp = get_cpp_module()
        dev = cpp.cpu_device() if device == 'cpu' else None
        tensor = cpp.Tensor.randn(list(shape), dev)
        return CppTensor._from_cpp_tensor(tensor)
    
    @staticmethod
    def arange(start: float, stop: float, step: float = 1.0, 
               device: str = 'cpu') -> 'CppTensor':
        """Create tensor with arange."""
        cpp = get_cpp_module()
        dev = cpp.cpu_device() if device == 'cpu' else None
        tensor = cpp.Tensor.arange(start, stop, step, dev)
        return CppTensor._from_cpp_tensor(tensor)
    
    # Operator overloads
    def __add__(self, other: Union['CppTensor', float]) -> 'CppTensor':
        cpp = get_cpp_module()
        if isinstance(other, CppTensor):
            result = cpp.add(self._tensor, other._tensor)
        else:
            # Scalar addition
            scalar_tensor = CppTensor.ones(self.shape, self.device)
            scalar_tensor._tensor.fill(float(other))
            result = cpp.add(self._tensor, scalar_tensor._tensor)
        return CppTensor._from_cpp_tensor(result)
    
    def __sub__(self, other: Union['CppTensor', float]) -> 'CppTensor':
        cpp = get_cpp_module()
        if isinstance(other, CppTensor):
            result = cpp.sub(self._tensor, other._tensor)
        else:
            scalar_tensor = CppTensor.ones(self.shape, self.device)
            scalar_tensor._tensor.fill(float(other))
            result = cpp.sub(self._tensor, scalar_tensor._tensor)
        return CppTensor._from_cpp_tensor(result)
    
    def __mul__(self, other: Union['CppTensor', float]) -> 'CppTensor':
        cpp = get_cpp_module()
        if isinstance(other, CppTensor):
            result = cpp.mul(self._tensor, other._tensor)
        else:
            result = cpp.scalar_mul(self._tensor, float(other))
        return CppTensor._from_cpp_tensor(result)
    
    def __truediv__(self, other: Union['CppTensor', float]) -> 'CppTensor':
        cpp = get_cpp_module()
        if isinstance(other, CppTensor):
            result = cpp.div(self._tensor, other._tensor)
        else:
            scalar_tensor = CppTensor.ones(self.shape, self.device)
            scalar_tensor._tensor.fill(float(other))
            result = cpp.div(self._tensor, scalar_tensor._tensor)
        return CppTensor._from_cpp_tensor(result)
    
    def __pow__(self, other: Union['CppTensor', float]) -> 'CppTensor':
        cpp = get_cpp_module()
        if isinstance(other, CppTensor):
            result = cpp.pow(self._tensor, other._tensor)
        else:
            scalar_tensor = CppTensor.ones(self.shape, self.device)
            scalar_tensor._tensor.fill(float(other))
            result = cpp.pow(self._tensor, scalar_tensor._tensor)
        return CppTensor._from_cpp_tensor(result)
    
    def __repr__(self) -> str:
        return f"CppTensor(shape={self.shape}, device={self.device})"
    
    def __array__(self) -> np.ndarray:
        """Support numpy array conversion."""
        return self.numpy()


# Tensor operation functions
def add(a: CppTensor, b: CppTensor) -> CppTensor:
    """Element-wise addition."""
    cpp = get_cpp_module()
    result = cpp.add(a._tensor, b._tensor)
    return CppTensor._from_cpp_tensor(result)


def sub(a: CppTensor, b: CppTensor) -> CppTensor:
    """Element-wise subtraction."""
    cpp = get_cpp_module()
    result = cpp.sub(a._tensor, b._tensor)
    return CppTensor._from_cpp_tensor(result)


def mul(a: CppTensor, b: CppTensor) -> CppTensor:
    """Element-wise multiplication."""
    cpp = get_cpp_module()
    result = cpp.mul(a._tensor, b._tensor)
    return CppTensor._from_cpp_tensor(result)


def div(a: CppTensor, b: CppTensor) -> CppTensor:
    """Element-wise division."""
    cpp = get_cpp_module()
    result = cpp.div(a._tensor, b._tensor)
    return CppTensor._from_cpp_tensor(result)


def matmul(a: CppTensor, b: CppTensor) -> CppTensor:
    """Matrix multiplication."""
    cpp = get_cpp_module()
    result = cpp.matmul(a._tensor, b._tensor)
    return CppTensor._from_cpp_tensor(result)


def relu(x: CppTensor) -> CppTensor:
    """ReLU activation."""
    cpp = get_cpp_module()
    result = cpp.relu(x._tensor)
    return CppTensor._from_cpp_tensor(result)


def sigmoid(x: CppTensor) -> CppTensor:
    """Sigmoid activation."""
    cpp = get_cpp_module()
    result = cpp.sigmoid(x._tensor)
    return CppTensor._from_cpp_tensor(result)


def tanh(x: CppTensor) -> CppTensor:
    """Tanh activation."""
    cpp = get_cpp_module()
    result = cpp.tanh(x._tensor)
    return CppTensor._from_cpp_tensor(result)


def softmax(x: CppTensor) -> CppTensor:
    """Softmax activation."""
    cpp = get_cpp_module()
    result = cpp.softmax(x._tensor)
    return CppTensor._from_cpp_tensor(result)


def exp(x: CppTensor) -> CppTensor:
    """Element-wise exponential."""
    cpp = get_cpp_module()
    result = cpp.exp(x._tensor)
    return CppTensor._from_cpp_tensor(result)


def log(x: CppTensor) -> CppTensor:
    """Element-wise natural logarithm."""
    cpp = get_cpp_module()
    result = cpp.log(x._tensor)
    return CppTensor._from_cpp_tensor(result)


def sum(x: CppTensor, axes: Optional[List[int]] = None, 
        keepdims: bool = False) -> CppTensor:
    """Sum reduction."""
    cpp = get_cpp_module()
    axes_list = axes if axes is not None else []
    result = cpp.sum(x._tensor, axes_list, keepdims)
    return CppTensor._from_cpp_tensor(result)


def mean(x: CppTensor, axes: Optional[List[int]] = None, 
         keepdims: bool = False) -> CppTensor:
    """Mean reduction."""
    cpp = get_cpp_module()
    axes_list = axes if axes is not None else []
    result = cpp.mean(x._tensor, axes_list, keepdims)
    return CppTensor._from_cpp_tensor(result)
