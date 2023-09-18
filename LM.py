import nltk
import string, json
import torch
import torchtext
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from math import exp as exponential
import matplotlib.pyplot as plt


class FWDataset(Dataset):
    def __init__(self, filename, seq_len, vocab) :
        super().__init__()
        # Preprocessing the text corpus
        with open(filename, 'r') as f:
            self.tokens = json.load(f)
        print(len(self.tokens))
        self.tokens = self.tokens[:(int(len(self.tokens) / 100))]
        
        # Loading pretrained embedding
        # self.pretrained_embedding = torchtext.vocab.GloVe(name='6B', dim=300)
        
        # Creating vocabulary
        self.vocab = vocab
        
        # Max sequence length
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, idx):
        # getting pretrained embedding for previous 5 words
        # X = self.pretrained_embedding.get_vecs_by_tokens(self.tokens[idx : idx + self.seq_len])
        X = [self.vocab[token] for token in self.tokens[idx : idx + self.seq_len]]
        X = torch.tensor(X)   
                
        # Target (6th word)
        y =  [self.vocab[token] for token in self.tokens[idx + 1 : idx + self.seq_len + 1]]
        y = torch.tensor(y)
        return X, y


class BWDataset(FWDataset):
    def __init__(self, filename, seq_len, vocab) :
        super().__init__(filename, seq_len, vocab)
        
    def __len__(self):
        return super().__len__(self)
    
    def __getitem__(self, idx):
        # getting forward dataset
        X, y = super().__getitem__(idx)
        
        # making target input and input as target.
        X, y = y, X
        
        # reversing the sequence
        X, y = torch.flip(X, [0]), torch.flip(y, [0])
        return X, y


def create_vacab(filename):
    # filenname is a json containgin a list of tokens
    with open(filename, 'r') as f:
        tokens = json.load(f)
    vocab = torchtext.vocab.build_vocab_from_iterator([[token] for token in tokens], min_freq=2, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


class ForwardLanguageModel(nn.Module):
    def __init__(self, vocab_size, pre_embedd):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(pre_embedd.vectors, freeze=True)
        self.lstm1 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.hidden2 = nn.Linear(50, vocab_size)
 
    def forward(self, x):
        # Lstm layers with 300 dim output
        x = self.embedding(x)
        x, hidden = self.lstm1(x)
        x, hidden = self.lstm2(x)
        
        # converts 300 dim vector into vocab_size dim vector
        x = self.hidden2(x)
        return x
    
    def forward2(self, x, w1):
        x1, hidden = self.lstm1(x)
        x2, hidden = self.lstm2(x1)
        w1 = w1
        w2 = 1 - w1
        return w1 * x1 + w2 * x2


def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for X, y in tqdm(dataloader):
        # data
        X, y = X.to(device), y.to(device)
        
        # forward pass
        pred = model(X)
        y = y.reshape(-1)
        pred = pred.reshape(-1, pred.shape[2])
        loss = loss_fn(pred, y)
        
        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_loop(dataloader, model, loss_fun, device):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            # getting data
            X, y = X.to(device), y.to(device)
            
            # forward pass
            pred = model(X)
            y = y.reshape(-1)
            pred = pred.reshape(-1, pred.shape[2])            
            
            # stats
            test_loss += loss_fun(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= len(dataloader)
    try:
        peprlexity = exponential(test_loss)
    except OverflowError:
        peprlexity = float('inf')
    correct /= (len(dataloader.dataset) * X.shape[1])
    return test_loss, correct, peprlexity


