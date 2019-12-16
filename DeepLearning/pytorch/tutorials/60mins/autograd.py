import torch

#torch.Tensor is the central class of the package. If you set its attribute .requires_grad as True, it starts to track all operations on it. When you finish your computation you can call .backward() and have all the gradients computed automatically. The gradient for this tensor will be accumulated into .grad attribute.

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2 #tensor([[3., 3.],[3., 3.]], grad_fn=<AddBackward0>)
z = y * y * 3
out = z.mean()

print(z, out) #tensor([[27., 27.],[27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)

## Gradients
out.backward()
print(x.grad) #tensor([[4.5000, 4.5000],[4.5000, 4.5000]])