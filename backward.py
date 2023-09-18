import nltk
import string, json
import torch
import torchtext
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from math import exp as exponential
import matplotlib.pyplot as plt

from LM import *


def plot_stats(stats):
    x = range(len(stats))
    loss = [stat[0] for stat in stats]
    accuracy = [stat[1] for stat in stats]
    perplexity = [stat[2] for stat in stats]
    
    plt.clf()
    plt.plot(x, loss, label='loss')
    plt.savefig('bwloss.png')
    
    plt.clf()
    plt.plot(x, accuracy, label='accuracy')
    plt.savefig('bwaccuracy.png')
    
    plt.clf()
    plt.plot(x, perplexity, label='perplexity')
    plt.savefig('bwperplexity.png')


if __name__ == "__main__":        
    # hyperparameters
    device = torch.device("cuda", index=0)
    batch_size = 20
    epcohs = 1
    seq_len = 2
   
    # Data creation
    # vocab = create_vacab("Dataset/LMTokenizedData/train.json")
    pre_embedd = torchtext.vocab.GloVe('6B', dim=50)
    vocab = torchtext.vocab.vocab(pre_embedd.stoi)
    vocab.set_default_index(0)

    
    train_data = BWDataset("Dataset/LMTokenizedData/train.json", seq_len, vocab)
    test_data = BWDataset("Dataset/LMTokenizedData/test.json", seq_len, vocab)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Model creation
    Model = ForwardLanguageModel(len(vocab), pre_embedd).to(device)
    
    # Training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.001)
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
        
    
    # Saving the model
    torch.save(Model.state_dict(), 'bwmodel.pt')
