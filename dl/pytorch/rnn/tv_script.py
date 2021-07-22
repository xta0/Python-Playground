# load in data
import helper
import numpy as np
import torch
import torch.nn as nn
from string import punctuation
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader

data_dir = './data/Seinfeld_Scripts.txt'

text = helper.load_data(data_dir)

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    return {
        '.': '||PERIOD||',
        ',': '||COMMA||',
        '"': '||QUOTATION_MARK||',
        ';': '||SEMICOLON||',
        '!': '||EXCLAMATION_MARK||',
        '?': '||QUESTION_MARK||',
        '(': '||LEFT_PAREN>||',
        ')': '||RIGHT_PAREN||',
        '-': '||DASH||',
        '\n': '||RETURN||',
    }


helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function
    features = []
    targets = []
    for idx, _ in enumerate(words):
        if idx+sequence_length >= len(words):
            break
        x = words[idx:idx+sequence_length]
        features.append(x)
        y = words[idx+sequence_length]
        targets.append(y)
    train_x = np.array(features)
    train_y = np.array(targets)
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    # return a dataloader
    return train_loader

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
print(token_dict)
print(int_text[:10])
print(list(vocab_to_int.values())[:10])
print(list(int_to_vocab.values())[:10])

class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
    
        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # define model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        batch_size = nn_input.size(0)
        x = self.embedding(nn_input)
        x,h = self.lstm(x, hidden)
        x = x.contiguous().view(-1, self.hidden_dim)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.view(batch_size, -1, self.output_size)
        x = x[:, -1]

        # return one batch of output word scores and the hidden state
        return x, h
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden


def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """

    # move data to GPU, if available
    if train_on_gpu:
        inp, target = inp.cuda(), target.cuda()
    
    # perform backpropagation and optimization
    h = tuple([each.data for each in hidden])
    rnn.zero_grad()
    output, h = rnn(inp, h)
    loss = criterion(output(), target)
    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    optimizer.step()
    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), h

def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn


# Data params
# Sequence Length
sequence_length = 10  # of words in a sequence
# Batch Size
batch_size = 100
# data loader - do not change
train_loader = batch_data(int_text, sequence_length, batch_size)
# Training parameters
# Number of Epochs
num_epochs = 10
# Learning Rate
learning_rate = 0.001
# Model parameters
# Vocab size
vocab_size = len(vocab_to_int)
# Output size
output_size = vocab_size
# Embedding Dimension
embedding_dim = 128
# Hidden Dimension
hidden_dim = 256
# Number of RNN Layers
n_layers = 2

# Show stats for every n number of batches
show_every_n_batches = 500

# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

# saving the trained model
helper.save_model('./trained_tv_script', trained_rnn)
print('Model Trained and Saved')