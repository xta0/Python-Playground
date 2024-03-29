import torch
import torch.nn as nn
import torch.nn.functional as F


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
model.load_state_dict(torch.load('model_cifar10.pt'))
model.eval()

example = torch.rand(1, 3, 32, 32)
scripted_model = torch.jit.trace(model, example)
print(type(scripted_model))
#for param in scripted_model.parameters():
#    print(type(param), param.size())
scripted_model.save('./cifar10.pt')
ops = torch.jit.export_opnames(scripted_model)
print(ops)
scripted_model._save_for_lite_interpreter('./cifar10.bc')

