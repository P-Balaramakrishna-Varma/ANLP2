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
        
        # Creating vocabulary
        self.vocab = vocab
        
        # Max sequence length
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, idx):
        # getting pretrained embedding for previous 5 words
        X = []
        for token in self.tokens[idx : idx + self.seq_len]:
            if token in self.vocab:
                X.append(self.vocab[token])
            else:
                X.append(self.vocab['<unk>'])
        X = torch.tensor(X)   
        # X = [token for token in self.tokens[idx : idx + self.seq_len]]
     
        # Target (6th word)
        y = []
        for token in self.tokens[idx + 1 : idx + self.seq_len + 1]:
            if token in self.vocab:
                y.append(self.vocab[token])
            else:
                y.append(self.vocab['<unk>'])
        y = torch.tensor(y)
        # y =  [token for token in self.tokens[idx + 1 : idx + self.seq_len + 1]]

        return X, y


class BWDataset(FWDataset):
    def __init__(self, filename, seq_len, vocab) :
        super().__init__(filename, seq_len, vocab)
        
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        # getting forward dataset
        X, y = super().__getitem__(idx)
        
        # making target input and input as target.
        X, y = y, X
        
        # reversing the sequence
        X, y = torch.flip(X, [0]), torch.flip(y, [0])
        # X, y = X[::-1] , y[::-1]
        return X, y


class ForwardLanguageModel(nn.Module):
    def __init__(self, vocab_size, global_embeddings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(global_embeddings, freeze=True)
        self.lstm1 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.hidden2 = nn.Linear(50, vocab_size)
 
    def forward(self, x):
        x = self.embedding(x)
        x, hidden = self.lstm1(x)
        x, hidden = self.lstm2(x)
        
        x = self.hidden2(x)
        return x
    
    def forward2(self, x, w1):
        assert(w1 >= 0 and w1 <= 1)
        x = self.embedding(x)
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


def load_glove(device): 
    pre_embedd = torchtext.vocab.GloVe('6B', dim=50)
    
    embd = pre_embedd.vectors
    # print(embd.shape)
    embd = embd.to(device)
    avg = embd.mean(dim=0).reshape(1, -1)
    embd = torch.cat((embd, avg), dim=0)
    # print(embd.shape)
    # print(embd[-1] == avg[-1])
    
    vocab_dic = pre_embedd.stoi
    vocab_dic['<unk>'] = len(vocab_dic)
    # print(embd[vocab_dic['<unk>']] == embd[-1])
    
    return embd.to("cpu"), vocab_dic
        

if __name__ == "__main__":
    device = "cuda:0"
    
    ## Pretainted embedding
    # global_embeddings, vocab_dic = load_glove(device)
    
    # model = ForwardLanguageModel(vocab_size=len(vocab_dic), global_embeddings=global_embeddings).to(device)
    # em = model.embedding(torch.tensor([vocab_dic['<unk>']]).to(device))
    # print(em == global_embeddings[-1].to(device))
   
    # test_data = FWDataset("Dataset/LMTokenizedData/test.json", 5, vocab_dic)
    # test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # for X, y in test_dataloader:
    #     print(X, "\n", y, "\n\n\n\n")