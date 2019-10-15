## Implement a one hidden layer nerual network using Pytorch

import torch
from torch import nn


import torch.nn.functional as F
class Network2(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        
        return x

# Create the network and look at it's text representation
model = Network()
output = model.forward(torch.ones(64,784))
model.save("model.pt")
print(output)
