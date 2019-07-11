import torch

## Multilayer neural network
torch.manual_seed(7)

# Features are 3 random normal variables
# shape: 1x3
features = torch.randn((1,3))

# Define the size of each layer in our network
n_input = features.shape[1] # number of input units match number of input features
n_hidden = 2 # number of hidden unites
n_output = 1 # number of output units

# weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden) #3x2
W2 = torch.randn(n_hidden, n_output) #2x1

# and bias terms for hidden and output layer
B1 = torch.randn(1, n_hidden)
B2 = torch.randn(1, n_output)

def activation(x):
    return 1 / (1+torch.exp(-x))

h = activation(torch.mm(features, W1)+B1)
y = activation(torch.mm(h,W2)+B2)