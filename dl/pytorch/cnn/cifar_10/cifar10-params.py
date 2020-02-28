import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.2)
        self.fc1   = nn.Linear(64*4*4,500)
        self.fc2   = nn.Linear(500,10)
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
model.load_state_dict(torch.load('model_cifar10.pt'))
state = model.state_dict() #ordered dict
conv1_w = state["conv1.weight"]
conv1_b = state["conv1.bias"]
print(conv1_w.shape) # NCWH -> 16x3x3x3 
print(conv1_b.shape) #[16]

conv2_w = state["conv2.weight"]
conv2_b = state["conv2.bias"]
print(conv2_w.shape) # NCWH -> 32x16x3x3 
print(conv2_b.shape) #[32]

conv3_w = state["conv3.weight"]
conv3_b = state["conv3.bias"]
print(conv3_w.shape) # NCWH -> 64x32x3x3 
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
np.savetxt('mpscnn_conv1_w.txt', [conv1_w.permute(0,2,3,1).contiguous().view(-1).numpy()],delimiter=',')
np.savetxt('mpscnn_conv1_b.txt', [conv1_b.numpy()],delimiter=',')
np.savetxt('mpscnn_conv2_w.txt', [conv2_w.permute(0,2,3,1).contiguous().view(-1).numpy()],delimiter=',')
np.savetxt('mpscnn_conv2_b.txt', [conv2_b.numpy()],delimiter=',')
np.savetxt('mpscnn_conv3_w.txt', [conv3_w.permute(0,2,3,1).contiguous().view(-1).numpy()],delimiter=',')
np.savetxt('mpscnn_conv3_b.txt', [conv3_b.numpy()],delimiter=',')
np.savetxt('mpscnn_fc1_w.txt', [fc1_w.view(-1).numpy()],delimiter=',')
np.savetxt('mpscnn_fc1_b.txt', [fc1_b.numpy()],delimiter=',')
np.savetxt('mpscnn_fc2_w.txt', [fc2_w.view(-1).numpy()],delimiter=',')
np.savetxt('mpscnn_fc2_b.txt', [fc2_b.numpy()],delimiter=',')