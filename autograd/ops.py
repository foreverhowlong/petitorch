import numpy as np
from .function import Function
from petitorch.tensor import Tensor


def register_method(name):
    """mount a function on the Tensor class dynamically"""
    def decorator(fn):
        setattr(Tensor, name, fn) 
        return fn
    return decorator

def unbroadcast(grad:np.ndarray, target_shape:int)->np.ndarray:
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
        a_shape = ctx.saved_meta['a_shape']
        b_shape = ctx.saved_meta['b_shape']
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
    
class MatMul(Function):
    @classmethod
    def forward(cls, ctx, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_tensor(a,b)
        # print(f"MatMul:sizeof a:{a.shape};size of b:{b.shape}")
        return a@b
    
    @classmethod
    def backward(cls, ctx, error: np.ndarray):
        a, b = ctx.saved_tensors
        grad_a = unbroadcast(error@b.T,a.shape)
        grad_b = unbroadcast(a.T@error,b.shape)
        return grad_a, grad_b
    
class Sum(Function):
    @classmethod
    def forward(cls, ctx, a: np.ndarray) -> int:
        return a.sum()
    
    @classmethod
    def backward(cls, ctx, error:np.ndarray):
        return error
@register_method("sum")
def _tensor_sum(self):
    return Sum.apply(self)


class Mean(Function):
    @classmethod
    def forward(cls, ctx, a:np.ndarray)->int:
        ctx.save_meta(n=a.size)
        return np.mean(a)
    @classmethod
    def backward(cls,ctx,error:np.ndarray)->int:
        n = ctx.saved_meta['n']
        return error/n
@register_method("mean")
def _tensor_mean(self):
    return Mean.apply(self)