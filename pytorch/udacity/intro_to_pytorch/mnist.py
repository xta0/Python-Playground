# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import helper
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


# Define a transform to normalize the data
transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

print(type(images)) #tensor
print(images.shape) #64,1,28,28
print(labels.shape) #64

# plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
# plt.show()

## Your solution
def activation(x):
    return 1/(1+torch.exp(-x))

#flatten the image
features = images.view(64,-1) #64x784

n_input  = features.shape[1]
n_hidden = 256
n_output = 10

W1 = torch.randn(n_input,  n_hidden) #784 x256
W2 = torch.randn(n_hidden, n_output) #256x10

B1 = torch.randn(1, n_hidden)
B2 = torch.randn(1, n_output)

h = activation(torch.mm(features, W1)+B1) #64x256
y = activation(torch.mm(h, W2)+B2) #64x10

print(y)

