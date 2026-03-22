import numpy as np
from .function import Function

def unbroadcast(grad, target_shape):
    """
    unbroadcast:turn grad back to the shape of target shape
    """
    if np.shape(grad) == target_shape:
        return grad
    
    ndims_added = grad.ndim - len(target_shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
        

    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
            
    return grad

class Add(Function):
    @classmethod
    def forward(cls, ctx, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_meta(a_shape=a.shape, b_shape=b.shape)
        return a + b

    @classmethod
    def backward(cls, ctx, error: np.ndarray):
        #get the target shape
        a_shape = ctx.saved_kwargs['a_shape']
        b_shape = ctx.saved_kwargs['b_shape']
        #make sure to return the target shape
        return unbroadcast(error, a_shape), unbroadcast(error, b_shape)

class Mul(Function):
    @classmethod
    def forward(cls, ctx, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_tensor(a, b)
        return a * b

    @classmethod
    def backward(cls, ctx, error: np.ndarray):
        a, b = ctx.saved_tensors
        # calculate the gradient, and reshape it
        grad_a = unbroadcast(b * error, a.shape)
        grad_b = unbroadcast(a * error, b.shape)
        return grad_a, grad_b