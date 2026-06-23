"""
Standard mathematical operations and their backward derivatives registry for the computational graph.
Provides forward/backward callables mapped to string identifiers to support serialization.
"""

import numpy as np
from typing import Dict, Tuple, Callable, List
from .backend import add, mul, matmul, relu, sigmoid, tanh, to_numpy, sum as backend_sum


class GraphOp:
    def __init__(
        self,
        name: str,
        forward_fn: Callable,
        backward_fn: Callable,
    ):
        self.name = name
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn


# Define forward & backward implementations
def add_forward(x, y):
    return add(x, y)


def add_backward(out_grad, inputs):
    x, y = inputs
    # Convert to numpy for broadcasting logic
    x_np = to_numpy(x)
    y_np = to_numpy(y)
    out_grad_np = to_numpy(out_grad)
    
    # Handle broadcasting by summing over broadcasted dimensions
    dx = out_grad
    dy = out_grad
    while dx.ndim > x.ndim:
        dx = backend_sum(dx, [0])
    for axis, dim in enumerate(x.shape):
        if dim == 1:
            dx = backend_sum(dx, [axis], True)
            
    while dy.ndim > y.ndim:
        dy = backend_sum(dy, [0])
    for axis, dim in enumerate(y.shape):
        if dim == 1:
            dy = backend_sum(dy, [axis], True)
            
    return [dx, dy]


def mul_forward(x, y):
    return mul(x, y)


def mul_backward(out_grad, inputs):
    x, y = inputs
    dx = mul(out_grad, y)
    dy = mul(out_grad, x)
    
    # Handle broadcasting
    while dx.ndim > x.ndim:
        dx = backend_sum(dx, [0])
    for axis, dim in enumerate(x.shape):
        if dim == 1:
            dx = backend_sum(dx, [axis], True)
            
    while dy.ndim > y.ndim:
        dy = backend_sum(dy, [0])
    for axis, dim in enumerate(y.shape):
        if dim == 1:
            dy = backend_sum(dy, [axis], True)
            
    return [dx, dy]


def matmul_forward(x, y):
    return matmul(x, y)


def matmul_backward(out_grad, inputs):
    x, y = inputs
    # out = x * y. d_out / dx = out_grad * y.T, d_out / dy = x.T * out_grad
    dx = matmul(out_grad, transpose(y))
    dy = matmul(transpose(x), out_grad)
    return [dx, dy]


def relu_forward(x):
    return relu(x)


def relu_backward(out_grad, inputs):
    x = inputs[0]
    x_np = to_numpy(x)
    out_grad_np = to_numpy(out_grad)
    dx = mul(out_grad, (x_np > 0.0))
    return [dx]


def sigmoid_forward(x):
    return sigmoid(x)


def sigmoid_backward(out_grad, inputs):
    x = inputs[0]
    sig = sigmoid_forward(x)
    sig_np = to_numpy(sig)
    one_minus_sig = sub(1.0, sig_np)
    dx = mul(out_grad, mul(sig_np, one_minus_sig))
    return [dx]


def tanh_forward(x):
    return tanh(x)


def tanh_backward(out_grad, inputs):
    x = inputs[0]
    t = tanh(x)
    t_np = to_numpy(t)
    one_minus_t2 = sub(1.0, mul(t_np, t_np))
    dx = mul(out_grad, one_minus_t2)
    return [dx]


def transpose(x):
    """Transpose a 2D tensor."""
    from .backend import Backend, get_backend
    if get_backend() == Backend.CPP:
        return x.transpose([1, 0])
    else:
        return x.T


# Register operations
OP_REGISTRY: Dict[str, GraphOp] = {
    "add": GraphOp("add", add_forward, add_backward),
    "mul": GraphOp("mul", mul_forward, mul_backward),
    "matmul": GraphOp("matmul", matmul_forward, matmul_backward),
    "relu": GraphOp("relu", relu_forward, relu_backward),
    "sigmoid": GraphOp("sigmoid", sigmoid_forward, sigmoid_backward),
    "tanh": GraphOp("tanh", tanh_forward, tanh_backward),
}
