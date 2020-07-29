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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


# helper function to un-normalize and display an image
def showSamples(images, labels, classes):
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(6):
        ax = fig.add_subplot(1, 6, idx + 1, xticks=[], yticks=[])
        ax.set_title(classes[labels[idx]])
        img = images[idx]
        img = img / 2 + 0.5
        img = img.numpy()
        plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


batch_size = 20
valid_size = 0.2  # use 20% of the training data as validation data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))
])
data_path = './cifar10_data/'
train_data = datasets.CIFAR10(data_path,
                              train=True,
                              download=True,
                              transform=transform)
test_data = datasets.CIFAR10(data_path,
                             train=False,
                             download=True,
                             transform=transform)

num_train = len(train_data)  #50000
indices = list(range(num_train))
np.random.shuffle(indices)
split_index = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split_index:], indices[:split_index]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

# specify the image classes
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]

# visualize training data
# images,labels = iter(train_loader).next() #images: [20, 3, 32, 32], label:[20]
# showSamples(images,labels,classes)


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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten the input
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x, dim=1)
        return x


model = Classifier()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

loss_fn = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 30

# # Start training
valid_loss_min = np.Inf  # track change in validation loss
train_loss_vec = []
valid_loss_vec = []
for epoch in range(1, epochs):
    train_loss = 0
    model.train()
    for images, labels in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    else:
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in valid_loader:
                if train_on_gpu:
                    images, labels = images.cuda(), labels.cuda()
                output = model(images)
                loss = loss_fn(output, labels)
                valid_loss += loss.item() * images.size(0)

        #calculate the average losses
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        train_loss_vec.append(train_loss)
        valid_loss_vec.append(valid_loss)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.
              format(epoch, train_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            print(
                'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                .format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'model_cifar10.pt')
            valid_loss_min = valid_loss

# Plot the train error
plt.plot(train_loss_vec, label='Training loss')
plt.plot(valid_loss_vec, label='Validation loss')
plt.legend(frameon=False)
plt.show()

# Save params
model.half()  #save "fp16"
state = model.state_dict()  #ordered dict
conv1_w = state["conv1.weight"]
conv1_b = state["conv1.bias"]
print(conv1_w.shape)  # NCHW -> 16x3x3x3
print(conv1_b.shape)  #[16]

conv2_w = state["conv2.weight"]
conv2_b = state["conv2.bias"]
print(conv2_w.shape)  # NCHW -> 32x16x3x3
print(conv2_b.shape)  #[32]

conv3_w = state["conv3.weight"]
conv3_b = state["conv3.bias"]
print(conv3_w.shape)  # NCHW -> 64x32x3x3
print(conv3_b.shape)  #[64]

fc1_w = state["fc1.weight"]
fc1_b = state["fc1.bias"]
print(fc1_w.shape)  # 500x1024
print(fc1_b.shape)  #[500]

fc2_w = state["fc2.weight"]
fc2_b = state["fc2.bias"]
print(fc2_w.shape)  # 10x500
print(fc2_b.shape)  #[10]

# save all those weights and bias
# MPSCNN requires [oC kH kW iC] -> NHWC
# PyTorch is NCHW
np.savetxt('./params/conv1_W.txt',
           [conv1_w.permute(0, 2, 3, 1).contiguous().view(-1).numpy()],
           delimiter=',')
np.savetxt('./params/conv1_b.txt', [conv1_b.numpy()], delimiter=',')
np.savetxt('./params/conv2_W.txt',
           [conv2_w.permute(0, 2, 3, 1).contiguous().view(-1).numpy()],
           delimiter=',')
np.savetxt('./params/conv2_b.txt', [conv2_b.numpy()], delimiter=',')
np.savetxt('./params/conv3_W.txt',
           [conv3_w.permute(0, 2, 3, 1).contiguous().view(-1).numpy()],
           delimiter=',')
np.savetxt('./params/conv3_b.txt', [conv3_b.numpy()], delimiter=',')
np.savetxt('./params/fc1_W.txt', [fc1_w.view(-1).numpy()], delimiter=',')
np.savetxt('./params/fc1_b.txt', [fc1_b.numpy()], delimiter=',')
np.savetxt('./params/fc2_W.txt', [fc2_w.view(-1).numpy()], delimiter=',')
np.savetxt('./params/fc2_b.txt', [fc2_b.numpy()], delimiter=',')
