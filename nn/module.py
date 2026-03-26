from petitorch.tensor import Tensor
from .parameter import Parameter
class Module:
    def __init__(self):
        pass
    def parameters(self):
        params = []
        for key,value in self.__dict__.items():
            if isinstance(value,Parameter):
                params.append(value)
            elif isinstance(value,Module):
                params.extend(value.parameters())
        return params
    def forward(self,*args,**kwargs):
        raise NotImplementedError
    def __call__(self,*args,**kwargs):
        return self.forward(*args,**kwargs)
    def zero_grad(self):
        for param in self.parameters():
            param.grad = None