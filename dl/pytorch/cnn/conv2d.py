import torch 
import torch.nn as nn
import numpy as np

#conv2d(input_channel, output_channel, kernel size, padding)
#(n+2p-f+1)x(n+2p-f+1)

def saveToMPSImage(x):
    padding = torch.ones([1,2,2])
    x       = torch.cat((x,padding),0) #[4,2,2] 
    x       = x.permute(1,2,0) #[32,32,4] [RGBA,RGBA,RGBA,...]
    x       = x.contiguous().view(-1).numpy()
    print(f"mps: {x}")
    np.savetxt('./x.txt',[x], delimiter=',')    
    
x  = torch.randn(4,2,2)
x = torch.tensor([[[1.0,1.0],[1.0,0]],[[1.0,1.0],[1.0,0]],[[1.0,1.0],[1.0,0]],[[1.0,1.0],[1.0,0]]])
# saveToMPSImage(x)
x  = x.unsqueeze(0) #[1,4,2,2]

# input: [1,4,2,2]
print('---------------------')
conv    = nn.Conv2d(4,1,2,padding=0,bias=False) #ic=3, oc=2, kernel size =2, padding=1
conv.weight = nn.Parameter(torch.ones([1,4,2,2]))
w       = conv.weight
print(w.shape)
wp      = w.permute(0,2,3,1).contiguous().view(-1).detach().numpy()
print(wp)
np.savetxt('./conv_W.txt',[wp], delimiter=',')
# b       = conv.bias.half()
# np.savetxt('./conv_b.txt',[b.detach().numpy()], delimiter=',')
print('---------------------')
y       = conv(x)
print(y.shape)
y       = y.view(-1).detach().numpy()
print(y)
np.savetxt('./y.txt',[y], delimiter=',')
