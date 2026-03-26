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
        self.saved_meta = {} 

    def save_tensor(self, *tensors):
        self.saved_tensors = tensors
        
    def save_meta(self, **kwargs):
        #if the original tensor is not saved, this can save the metadata
        self.saved_meta.update(kwargs)
        
class NoOpContext:
    def save_tensor(self,*ndarrays):
        pass
    def save_meta(self,**kwargs):
        pass

#Function doesn't store any info. the apply() method instanciate context, which consititute the compute graph
#the base class of all the operators
class Function:
    
    #The most important function in constructing the compute graph
    #all of the operations(+ * @,...) regarding Tensor are registered into the compute graph here
    @classmethod
    def apply(cls,*args)-> Tensor: 
        tensors = [arg for arg in args if isinstance(arg,Tensor)]
        ndarrays = list(map(attrgetter('data'), tensors))
        requires_grad = any(t.requires_grad for t in tensors)
        if requires_grad:
            ctx = Context()
        else:
            #if we don't need grad, instanciate this dummy context
            #so that we can reuse our forward() method.
            ctx = NoOpContext()
        result = Tensor(cls.forward(ctx,*ndarrays),requires_grad=requires_grad)
        if not requires_grad:
            return result
        #the edge of the compute graph.
        ctx.prev_edges = tensors
        ctx.backward_op = cls
        result.grad_fn = ctx
        return result
    
    
    #this two methods should be implemented in its children class.
    @classmethod
    def forward(cls,ctx: Context,*ndarrays)-> np.ndarray:
        raise NotImplementedError
    
    @classmethod
    def backward(cls,ctx,error:np.ndarray) ->np.ndarray:
        raise NotImplementedError
    