import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
from torch import optim
import os
import numpy as np
import sys
sys.path.append('../')
import helper


os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Construct NN
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        # self.fc2 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(256, 10)
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        # output so no dropout here
        x = F.log_softmax(self.fc2(x), dim=1)

        return x

model = Classifier()
print(model)
dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[3].view(images[1].shape[0], -1)
np.savetxt('input-grayscale-img.txt',[img.view(-1).numpy()],delimiter=',')
# print(img.shape) # [1, 784]
# ps = torch.exp(model(img))
# helper.view_classify(img, ps, version="MNIST")

## Train the network
loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
epochs = 30
train_losses, test_losses = [],[]
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images) # [1, 784]
        loss = loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        test_loss = 0 
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                log_ps = model(images)
                test_loss += loss_fn(log_ps, labels) #计算test_loss
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(top_class.shape[0],-1) 
                accuracy += torch.mean(equals.type(torch.FloatTensor)) #计算accuracy
        model.train()
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

import matplotlib.pyplot as plt
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[1].view(images[1].shape[0], -1)
output = model(img)
print(output.shape)
ps = torch.exp(output)
print(ps)
helper.view_classify(img, ps, version='MNIST')

# save the model
model.half()
torch.save(model.state_dict(), 'model_mnist.pt')

# Save parameters
state = model.state_dict()
print("The state dict keys: \n\n", state.keys())

fc1_w = state["fc1.weight"] 
fc1_b = state["fc1.bias"]
print(fc1_w.shape) # 500x1024
print(fc1_b.shape) #[500]

fc2_w = state["fc2.weight"] 
fc2_b = state["fc2.bias"]
print(fc2_w.shape) # 10x500
print(fc2_b.shape) #[10]

# save all those weights and bias
np.savetxt('mpscnn_mnist_fc1_w.txt', [fc1_w.view(-1).numpy()],delimiter=',')
np.savetxt('mpscnn_mnist_fc1_b.txt', [fc1_b.numpy()],delimiter=',')
np.savetxt('mpscnn_mnist_fc2_w.txt', [fc2_w.view(-1).numpy()],delimiter=',')
np.savetxt('mpscnn_mnist_fc2_b.txt', [fc2_b.numpy()],delimiter=',')





