# Petitorch

petitorch is a minimalistic, automatic differentiation framework built entirely from scratch using pure Python and NumPy.

It is designed to replicate the underlying architecture of PyTorch, specifically focusing on the dynamic computational graph (Define-by-Run), the dynamic dispatcher mechanism, and the Adjoint Method (Vector-Jacobian Product) for reverse-mode automatic differentiation (gradient pull-back).

## Architecture and Module Implementation

The framework is strictly decoupled into several core components: data containers, the autograd engine, mathematical operators, and high-level neural network abstractions.
### 1. Core Data Structure (tensor.py)

The Tensor class serves as the fundamental data carrier. It is intentionally kept lightweight and does not implement mathematical logic directly.

    State Management: It encapsulates the underlying NumPy array (data), the gradient array (grad), a boolean flag (requires_grad), and a reference to the operation that created it (grad_fn).

    Syntactic Sugar: It overrides Python magic methods (e.g., __add__, __mul__, __matmul__) to route mathematical operations to the underlying autograd dispatcher, acting purely as a proxy.

 ### 2. Autograd Engine (autograd/)

This directory contains the core logic for building and traversing the Directed Acyclic Graph.

    The Dispatcher (function.py):

        Contains the Function base class. Its apply method acts as a dynamic dispatcher. During the forward pass, it inspects the requires_grad flags of the input tensors to determine if the autograd engine should be engaged.

        Implements the Context class, which represents a node in the computational graph. It stores forward-pass tensors (saved_tensors) and metadata (saved_meta) necessary for calculating derivatives. It also records the upstream dependencies (prev_edges) to link the graph.

        Implements Zero-Overhead Inference: If no inputs require gradients, the dispatcher instantiates a NoOpContext, bypassing graph construction and memory allocation entirely.

    The Backward Engine (engine.py):

        Initiated by calling tensor.backward(). It enforces that gradients can only be implicitly created for scalar outputs.

        Performs a Topological Sort using Depth-First Search starting from the root tensor to ensure gradients are computed in the correct dependency order.

        Iterates through the sorted nodes in reverse, invoking the adjoint operator (backward method) of each function. It meticulously handles multi-branch gradient accumulation by storing intermediate gradients in a dictionary and accumulating them into the .grad attributes of leaf tensors.

    Mathematical Operators (ops.py):

        Concrete implementations of mathematical operations (e.g., Add, Mul, MatMul, Sum).

        Each operator defines a forward method (executing pure NumPy operations) and a backward method (computing the Vector-Jacobian Product).

        Implements an unbroadcast utility function to handle NumPy's implicit broadcasting. This ensures that gradients flowing backward are correctly summed and reduced to match the original shapes of the input tensors.

### 3. Neural Network API (nn/)

A high-level wrapper to facilitate the construction of neural network architectures.

    Parameter (parameter.py): A subclass of Tensor that enforces requires_grad=True upon initialization, representing learnable weights.

    Module (module.py): The base class for all neural network layers. It utilizes Python's reflection (__dict__) to recursively discover and collect all Parameter instances and sub-modules attached to it.

    Linear (linear.py): A standard fully connected layer implementation managing weight and bias parameters, initialized using standard deviation scaling.

### 4. Optimization (optim/)

Decoupled entirely from the computational graph, the optimizer is responsible for updating the numerical values of the parameters.

    SGD (optimizer.py): Implements standard Gradient Descent. It iterates through the provided parameters, updating their .data attributes using the computed .grad arrays and a specified learning rate. It strictly operates on the .data level to avoid triggering the autograd graph construction during weight updates. It also provides the zero_grad method to clear accumulated gradients between iterations.


## Examples you can try

There are some examples you can run to test the Petitorch.

    autograd_example.py: We construct a simple compute graph here, to test if the autograd engine can correctly perform Topological Sort and distribute gradient back the graph.

    training_example.py: We build a model MyModel, containing a linear layer. The program trains this model on generated data to mimic y = 3 * x + 2. This example shows that the Petitorch is functioning correctly.

## Future Work / To Be Implemented

While the core autograd engine and linear regression capabilities are fully functional, the framework requires further expansion to support modern deep learning models:

    Non-linear Activation Functions: Implementation of ReLU, Sigmoid, and Tanh operators in ops.py.

    Loss Functions: Implementation of CrossEntropyLoss and Softmax for classification tasks.

    Advanced Optimizers: Implementation of momentum-based optimizers such as Adam and RMSprop to mitigate valley oscillation during training.

    Tensor Reshaping: Support for operations like reshape, transpose, and view, along with their corresponding backward passes.

    Broadcasting Robustness: Further refinement of the unbroadcast mechanism to support more complex, multi-dimensional tensor broadcasting scenarios.