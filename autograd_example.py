import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from petitorch.tensor import Tensor

def test_engine():
    print("=========================================")
    print("Petitorch Autograd Engine Test")
    print("=========================================")

    # Initialize leaf node (which is a scalar Tensor)
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    #Test if the engine will break when dealing tensor whose requires_grad is False
    c_const = Tensor([4.0])

    print(f"Initial input: a={a.data}, b={b.data}, c_const={c_const.data}")

    #construct a tricky compute graph   
    term1 = a * b
    term2 = a * a
    loss = term1 + term2 + c_const

    print(f"loss: {loss.data} (expected: [14.])")
    
    # Backward Pass
    print("executing loss.backward()...")
    loss.backward()

    # Math deductions:
    # loss = a*b + a^2 + c
    # d(loss)/da = b + 2a = 3.0 + 2*(2.0) = 7.0
    # d(loss)/db = a = 2.0
    # d(loss)/dc = 0 
    
    print("\n=========================================")
    print("Gradient Check")
    print("=========================================")
    print(f"grad of a: {a.grad} \t| expected: [7.]")
    print(f"grad of b: {b.grad} \t| expected: [2.]")
    print(f"grad of c_const: {c_const.grad} \t| expected: None")
    
    if np.allclose(a.grad, [7.0]) and np.allclose(b.grad, [2.0]):
        print("The autograd engine works perfectly!")
    else:
        print("There's something wrong with the gradient!")

if __name__ == "__main__":
    test_engine()