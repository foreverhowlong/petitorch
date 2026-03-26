import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from petitorch.tensor import Tensor
import petitorch.nn as nn
from petitorch.optim import SGD

'''
This example shows how to train a linear model to learn y = 3 * x + 2
'''


#(batch,in_feature)
X_data = np.random.randn(100,1)
y_data = 3 * X_data + 2 + np.random.randn(1) * 0.1 # a little noise

# define our Model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        #(in_feature,out_feature)
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)

model = MyModel()

lr = 0.1
optimizer = SGD(model.linear.parameters(), lr)
print(f"learning rate = {lr}")


# training loop
for epoch in range(50):
    # turn numpy data into Tensor
    #requires_grad is False by default
    inputs = Tensor(X_data)
    targets = Tensor(y_data)
    
    # forward
    predictions = model(inputs)
    

    
    #  MSE Loss: sum((pred - target)^2) / N

    diff = predictions + (targets * Tensor([-1.0])) # pred - target
    loss = (diff * diff).mean()
    

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")


#print out the results
print("Weight W:", model.linear.weight.data)
print("Bias b:", model.linear.bias.data)