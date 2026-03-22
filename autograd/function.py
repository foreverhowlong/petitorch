import numpy as np
from petitorch.tensor import Tensor
from operator import attrgetter

#all the information needed for backprop is saved in context
class Context:
    def __init__(self):
        self.saved_tensors = ()
        self.prev_edges = []
        self.backward_op = None
        # used for saving meta data like shape
        self.saved_kwargs = {} 
        
    def save_tensor(self, *tensors):
        self.saved_tensors = tensors
        
    def save_meta(self, **kwargs):
        #if the original tensor is not saved, this can save the metadata
        self.saved_kwargs.update(kwargs)
    
class NoOpContext:
    def save_tensor(self,*ndarrays):
        pass
    def save_meta(self,**kwargs):
        pass

#Function doesn't store any info. the apply method instanciate context, which consititute the compute graph
class Function:
    
    @classmethod
    def apply(cls,*args)-> Tensor: 
        tensors = [arg for arg in args if isinstance(arg,Tensor)]
        ndarrays = list(map(attrgetter('data'), tensors))
        requires_grad = any(t.requires_grad for t in tensors)
        if requires_grad:
            ctx = Context()
        else:
            ctx = NoOpContext()
        result = Tensor(cls.forward(ctx,*ndarrays),requires_grad=requires_grad)
        if not requires_grad:
            return result
        ctx.prev_edges = tensors
        ctx.backward_op = cls
        result.grad_fn = ctx
        return result
        
    @classmethod
    def forward(cls,ctx: Context,*ndarrays)-> np.ndarray:
        raise NotImplementedError
    
    @classmethod
    def backward(cls,ctx,error:np.ndarray) ->np.ndarray:
        raise NotImplementedError
    