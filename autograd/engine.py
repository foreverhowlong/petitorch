# petitorch/autograd/engine.py
import numpy as np
from petitorch.tensor import Tensor

def backward(root_tensor:Tensor):
    """
    traverse the compute graph DAG and make the gradient flow.
    loss.backward() = backward(loss)
    """
    if root_tensor.data.size > 1:
        raise RuntimeError("grad can be implicitly created only for scalar outputs")
    
    # if the root_tensor doesn't need grad, just return
    if not root_tensor.requires_grad:
        return
    #grad_dict is used to store grad for intermediate node
    #key:Context value:np.ndarray
    grad_dict={}
    # initialize the grad of the starting point
    grad_dict[root_tensor.grad_fn] = np.ones_like(root_tensor.data)
    
    #topological sort: make sure that the error has been completed before calling its backwards
    topo_order = []
    def topo_sort(ctx, visited=None):
    
        """
        Performs topological sorting using Depth-First Search (DFS) and the three-color marking method.

        Args:
            ctx: The current node being visited.
            visited: A dictionary tracking the visitation state of nodes.
                    Not in dictionary (or 0) -> Unvisited
                    State 1 -> Currently being visited in the recursion stack (Visiting)
                    State 2 -> Finished visiting all predecessors and added to the stack (Visited)
            topo_order: A list storing the final topological sequence.
        """

        if visited is None:
            visited = {}

        #check the current state
        state = visited.get(ctx, 0)
        
        if state == 1:
            # if we bump into a node that's in the recurrence stack, there must be a loop
            raise ValueError(f"Circular Dependency! '{ctx}' forms a loop.Topological sort can only work on DAG")
        
        if state == 2:

            return topo_order
        #visiting (in the recursion stack)
        visited[ctx] = 1

        for edge_tensor in ctx.prev_edges:
            #skip leaf node
            if edge_tensor.grad_fn is not None: 
                topo_sort(edge_tensor.grad_fn,visited)    
                
        #visited
        visited[ctx] = 2
        topo_order.append(ctx)
        return topo_order
    topo_sort(root_tensor.grad_fn)
    
    
    for ctx in reversed(topo_order):
        # get the error of current node
        current_error = grad_dict[ctx]
        # call the backward() of this function
        #apply the J* operator to pullback gradient
        grads = ctx.backward_op.backward(ctx, current_error)
        # we want to make sure the data type is tuple
        if not isinstance(grads, tuple):
            grads = (grads,)
            
        # distribute grad
        for edge_tensor, grad in zip(ctx.prev_edges, grads):
            
            # None indicates no need for grad.
            if grad is None:
                continue
            if not edge_tensor.requires_grad:
                continue
            # A: leaf node
            # only leaf node has the feature:requires_grad=True，grad_fn=None
            if edge_tensor.grad_fn is None:
                # AccumulateGrad 
                if edge_tensor.grad is None:
                    edge_tensor.grad = grad
                else:
                    edge_tensor.grad += grad
                    
            # B: intermeidate node
            else:
                next_ctx = edge_tensor.grad_fn
                if next_ctx in grad_dict:
                    grad_dict[next_ctx] += grad
                else:
                    grad_dict[next_ctx] = grad
