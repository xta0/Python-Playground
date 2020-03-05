import torch 
import torch.nn as nn

## conv2d

#ncwh
x = torch.randn(1,3,32,32)
#input_channel, output_channel, kernel size, padding
#(n+2p-f+1)x(n+2p-f+1)
conv1 = nn.Conv2d(3,16,3,1)
y = conv1(x)
print(y.shape) #(1,16,30,30) nchw
