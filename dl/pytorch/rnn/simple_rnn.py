import torch
from torch import nn
import numpy as np

seq_len = 20

time_steps = np.linspace(0, np.pi, seq_len + 1)
data = np.sin(time_steps)  # (21,)
data.resize((seq_len+1, 1)) #

x = data[:-1]
y = data[1:]

print("x: ", x)
print("y: ", y)

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x, hidden):
        batch_Size = x.size(0)
        r_out, hidden = self.rnn(x, hidden)
        r_out = r_out.view(-1, self.hidden_dim)
        output = self.fc(r_out)
        return output, hidden
        
        