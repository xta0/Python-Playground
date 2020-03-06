import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# visualize training data
# images,labels = iter(train_loader).next() #images: [20, 3, 32, 32], label:[20]
# showSamples(images,labels,classes)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.2)
        self.fc1   = nn.Linear(64*4*4,128)
        self.fc2   = nn.Linear(128,10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #flatten the input
        x = x.view(-1,64*4*4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x, dim=1)
        return x

model = Classifier()
print(model)
model.load_state_dict(torch.load('model_cifar10.pt'))
model.eval()

# Save params
model.half() #save "fp16"
state = model.state_dict() #ordered dict
conv1_w = state["conv1.weight"]
conv1_b = state["conv1.bias"]
print(conv1_w.shape) # NCHW -> 16x3x3x3 
print(conv1_b.shape) #[16]

conv2_w = state["conv2.weight"]
conv2_b = state["conv2.bias"]
print(conv2_w.shape) # NCHW -> 32x16x3x3 
print(conv2_b.shape) #[32]

conv3_w = state["conv3.weight"]
conv3_b = state["conv3.bias"]
print(conv3_w.shape) # NCHW -> 64x32x3x3 
print(conv3_b.shape) #[64]

fc1_w = state["fc1.weight"] 
fc1_b = state["fc1.bias"]
print(fc1_w.shape) # 500x1024
print(fc1_b.shape) #[500]

fc2_w = state["fc2.weight"] 
fc2_b = state["fc2.bias"]
print(fc2_w.shape) # 10x500
print(fc2_b.shape) #[10]

# save all those weights and bias
# MPSCNN requires weights to be [oC kH kW iC] -> NHWC
# PyTorch is NCHW
np.savetxt('./params/conv1_W.txt', [conv1_w.permute(0,2,3,1).contiguous().view(-1).numpy()],delimiter=',')
np.savetxt('./params/conv1_b.txt', [conv1_b.numpy()],delimiter=',')
np.savetxt('./params/conv2_W.txt', [conv2_w.permute(0,2,3,1).contiguous().view(-1).numpy()],delimiter=',')
np.savetxt('./params/conv2_b.txt', [conv2_b.numpy()],delimiter=',')
np.savetxt('./params/conv3_W.txt', [conv3_w.permute(0,2,3,1).contiguous().view(-1).numpy()],delimiter=',')
np.savetxt('./params/conv3_b.txt', [conv3_b.numpy()],delimiter=',')
np.savetxt('./params/fc1_W.txt', [fc1_w.view(-1).numpy()],delimiter=',')
np.savetxt('./params/fc1_b.txt', [fc1_b.numpy()],delimiter=',')
np.savetxt('./params/fc2_W.txt', [fc2_w.view(-1).numpy()],delimiter=',')
np.savetxt('./params/fc2_b.txt', [fc2_b.numpy()],delimiter=',')

