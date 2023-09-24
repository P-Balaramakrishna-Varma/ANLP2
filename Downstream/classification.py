import string, json
import torch
import torchtext
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './ELMOv/')
from LM import load_glove
from elmov import Elmov

class ClassificationDataset(Dataset):
    def __init__(self, file_path, vocab):
        super().__init__()
        
        # storing the vocabulary
        self.vocab = vocab
        
        # self.data is a list of samples. Each sample is list of tokens, index.
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        # print(json.dumps(self.data, indent=4))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) :
        y = int(self.data[index][1]) - 1     ## makes a difference during cross entropy loss
        tokens = self.data[index][0]
        X = []
        for token in tokens:
            if token in self.vocab:
                X.append(self.vocab[token])
            else:
                X.append(self.vocab['<unk>'])
        # return tokens, y
        return torch.tensor(X), torch.tensor(y)
    
    
def custom_collate(batch):
    Sentences = [sample[0] for sample in batch]
    Classes = torch.tensor([sample[1] for sample in batch])

    Sentences = pad_sequence(Sentences, batch_first=True)
    return Sentences, Classes


class ClassificationModel(nn.Module):
    def __init__(self, forward_file, backward_file, len_vocab, gobal_embedding, w1=None):
        super().__init__()
        self.pre_trained = Elmov(forward_file, backward_file, len_vocab, gobal_embedding, w1)
        self.meaning_aggr = nn.LSTM(input_size=200, hidden_size=200, batch_first=True)
        self.hidden = nn.Linear(200, 30)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(30, 4)
    
    def forward(self, x):
        # word embedding
        x = self.pre_trained(x)
        
        # meaning aggregation
        x, hidden = self.meaning_aggr(x)
        meaning = hidden[0].reshape((x.shape[0], -1))
        # print(meaning.shape)
        
        # MLP classifier
        hidden_space = self.hidden(meaning)
        hidden_space = self.relu(hidden_space)
        logits = self.classifier(hidden_space)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for X, y in tqdm(dataloader):
        # data
        X, y = X.to(device), y.to(device)
        
        # forward pass
        pred = model(X)
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
            
            # stats
            test_loss += loss_fun(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    return test_loss, correct


def plot_stats(stats):
    stats = list(zip(*stats))
    x = range(len(stats[0]))
    
    plt.clf()
    plt.plot(x, stats[0], label='Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('./ClassificationLoss.png')

    plt.clf()
    plt.plot(x, stats[1], label='Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('./ClassificationAcc.png')



if __name__ == "__main__":        
    # hyperparameters
    device = torch.device("cuda", index=0)
    batch_size = 100
    epcohs = 2
    lr = 0.00001
   
    # Data creation
    global_embedding, vocab = load_glove(device)

    
    train_data = ClassificationDataset("./Dataset/AGTokenizedData/train.json", vocab)
    test_data = ClassificationDataset("./Dataset/AGTokenizedData/test.json", vocab)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    # Model creation
    Model = ClassificationModel('./fwmodel.pt', './bwmodel.pt', len(vocab), global_embedding).to(device)
    
    # Training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model.parameters(), lr=lr)
    stats = []
    for epoch in tqdm(range(epcohs)):
        train_loop(train_dataloader, Model, loss_fn, optimizer, device)
        stats.append(test_loop(test_dataloader, Model, loss_fn, device))
        print(stats[-1])
    plot_stats(stats)
    print(stats)
    
    # Testing
    Results = test_loop(test_dataloader, Model, loss_fn, device)
    print(Results)