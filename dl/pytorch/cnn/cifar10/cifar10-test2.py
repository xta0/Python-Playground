from torchvision import datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

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
model.eval()
print(model)

# Test
model.load_state_dict(torch.load('model_cifar10.pt'))
model.eval()

# load datasets 
rawset = datasets.CIFAR10('./cifar10_data/')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))
])

# sample  = rawset[0] #(img, label)
# img     = sample[0]
# l = sample[1]


def saveToMPSImage(img,label,idx):
    print(img)
    padding = torch.ones([1,32,32])
    rgba    = torch.cat((img,padding),0) #[4,32,32] [R][G][B][A]
    # print(rgba.shape)
    # mpsi    = rgba.permute(1,2,0) #[32,32,4] [RGBA,RGBA,RGBA,...]
    # plt.imshow(mpsi)
    # plt.show()
    mpsi    = mpsi.contiguous().view(-1).numpy()
    # print(mpsi.shape)
    np.savetxt(f'./mps/{idx}_{label}.txt', [mpsi], delimiter=',')

# load datasets 
rawset = datasets.CIFAR10('./cifar10_data/')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))
])

cnt = 1
cn  = 0
with torch.no_grad():
    for idx, (img, label) in enumerate(rawset):
        if idx < cnt:
            t = transform(img) #[1,28,28]
            saveToMPSImage(t,label,idx)
            t = t.unsqueeze(0)
            output = model(t)
            ps = torch.exp(output)
            p, clz = ps.topk(1, dim=1)
            clz = clz.item()
            if clz == label:
                print(f"True: ( output:{clz}, label:{label} )")
                cn += 1
            else:
                print(f"False: ( output:{clz}, label:{label} )")
    print(f"accuracy: {float(cn)/float(cnt)}")
