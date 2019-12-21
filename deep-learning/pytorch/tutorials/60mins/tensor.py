from __future__ import print_function
import torch

#Construct a randomly initialized matrix:
x = torch.rand(5, 3)
print(x)
print(x.shape) #[5,3] 5行3列

#Construct a matrix filled zeros and of dtype long:
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

#Construct a tensor directly from data:
x = torch.tensor([5.5, 3])
print(x.shape)
print(x.size())

#ops
x = torch.ones(5,3)
y = torch.rand(5,3)
print(x+y)
y.add_(x) #in-place operation

#Resizing: If you want to resize/reshape tensor, you can use torch.view:
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8) ## the size -1 is inferred from other dimensions
print(x.size(),y.size(),z.size())
