from torchvision import datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# specify the image classes
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]


def convertToMPSImage(x):
    shape = x.shape
    print("input shape: ", shape)
    slices = []
    n = int((shape[0] + 3) / 4)  # slices
    l = int(shape[0])
    d = l
    if l % 4 != 0:
        d = (int(l / 4) + 1) * 4
    # print("(slice, c, dst_c)", n, l, d)
    for i in range(n):
        if i * 4 + 4 < l:
            s = x[i * 4:i * 4 + 4]
            slices.append(s)
        else:
            # add padding
            padding = torch.zeros(1, shape[1], shape[2])
            s = x[i * 4:shape[0]]
            while l < d:
                # print("concating")
                s = torch.cat((s, padding), 0)
                l = l + 1
            slices.append(s)
    # print(slices)
    # flatten the numpy array
    slices = [
        x.permute(1, 2, 0).contiguous().view(-1).detach().numpy()
        for x in slices
    ]
    slices = np.concatenate(slices)
    return slices


def saveTempResult(t, name):
    if "fc" in name or name == "softmax":
        t = t.view(-1).detach().numpy()
        np.savetxt(f'./mps/{name}.txt', [t], delimiter=',')
    else:
        t1 = t.squeeze(0)
        t1 = convertToMPSImage(t1)
        np.savetxt(f'./mps/{name}_nhwc.txt', [t1], delimiter=',')
        t2 = t.view(-1).detach().numpy()
        np.savetxt(f'./mps/{name}_nchw.txt', [t2], delimiter=',')


def saveToMPSImage(img, label, idx):
    np.savetxt(f'./mps/orig_{idx}_{label}.txt', [img.view(-1).numpy()],
               delimiter=',')
    img = convertToMPSImage(img)
    np.savetxt(f'./mps/{idx}_{label}.txt', [img], delimiter=',')


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.conv1(x)
        saveTempResult(x, "conv1")
        x = F.relu(x)
        saveTempResult(x, "relu1")
        x = self.pool(x)
        saveTempResult(x, "pool1")

        x = self.conv2(x)
        saveTempResult(x, "conv2")
        x = F.relu(x)
        saveTempResult(x, "relu2")
        x = self.pool(x)
        saveTempResult(x, "pool2")

        x = self.conv3(x)
        saveTempResult(x, "conv3")
        x = F.relu(x)
        saveTempResult(x, "relu3")
        x = self.pool(x)
        saveTempResult(x, "pool3")
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x))) # [1,64,4,4]
        #flatten the input
        # print(x.shape)
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        saveTempResult(x, "fc1")
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        saveTempResult(x, "fc2")
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

sample = rawset[0]  #(img, label)
img = sample[0].convert('RGB')
img.save("cifar_10.png")
t = transform(img).unsqueeze(0)
output = model(t)
print(output)
ps = torch.exp(output)
print(ps)

# cnt = 10
# cn = 0
# with torch.no_grad():
#     for idx, (img, label) in enumerate(rawset):
#         if idx < cnt:
#             t = transform(img)
#             saveToMPSImage(t, label, idx)
#             t = t.unsqueeze(0)
#             output = model(t)
#             ps = torch.exp(output)
#             p, clz = ps.topk(1, dim=1)
#             clz = clz.item()
#             if clz == label:
#                 print(f"True: ( output:{clz}, label:{label}, prob:{p} )")
#                 cn += 1
#             else:
#                 print(f"False: ( output:{clz}, label:{label}, prob:{p} )")
#     print(f"accuracy: {float(cn)/float(cnt)}")
