import numpy as np
class Tensor:
    def __init__(self,data, grad: np.ndarray =None,  grad_fn = None,requires_grad = False):
        self.data = np.array(data,dtype=np.float32)
        self.grad = grad
        #grad_fn stores a context
        self.grad_fn = grad_fn
        self.requires_grad = requires_grad

    
    #make debugging easier
    def __repr__(self):
        return f"Tensor:{self.data},grad_fn:{self.grad_fn}"
    
    #wrapper to enable natural writing.
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other) 
        from .autograd.ops import Add
        return Add.apply(self, other)
    def __mul__(self,other):
        from .autograd.ops import Mul
        return Mul.apply(self, other)
    def __matmul__(self,other):
        from .autograd.ops import MatMul
        return MatMul.apply(self, other)
    
    #wrapper to trigger the backprop engine
    def backward(self):
        from .autograd.engine import backward
        backward(self)
    
        