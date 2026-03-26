import numpy as np
from petitorch.tensor import Tensor
from petitorch.nn.parameter import Parameter

class Optimizer():
    #register all the parameter of the module to optimizer
    def __init__(self,params:Parameter,*args,**kwargs):
        self.params = params
        
    def step(self):
        raise NotImplementedError
    
#It is called SGD, but what it does is just Param -= lr*grad.
#So it can also be used for BGD/Mini-match SGD
class SGD(Optimizer):
    def __init__(self,params:Parameter,lr:float = 0.01):
        super().__init__(params)
        self.lr = lr
        
    def step(self):
        for p in self.params:
            p.data = p.data - self.lr * p.grad
            
    def zero_grad(self):
        for p in self.params:
            p.grad = None