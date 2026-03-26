from .module import Module
from .parameter import Parameter
import numpy as np

class Linear(Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        stdv = 1. / np.sqrt(in_features)
        self.weight = Parameter(np.random.uniform(-stdv, stdv, (in_features, out_features)))
        self.bias = Parameter(np.zeros((1,out_features)))
        
    def forward(self,x):
        #X*W.T This is for cache locality
        return x@self.weight +self.bias