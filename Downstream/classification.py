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
        y = int(self.data[index][1])
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


if __name__ == "__main__":
    device = "cuda:0"
    global_embeddin, vocab = load_glove(device)
    dataset = ClassificationDataset('./Dataset/AGTokenizedData/test.json', vocab)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=custom_collate)
    Model = ClassificationModel('./fwmodel.pt', './bwmodel.pt', len(vocab), global_embeddin).to(device)
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        Model(X)