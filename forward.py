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
    plt.savefig('fwloss.png')
    
    plt.clf()
    plt.plot(x, accuracy, label='accuracy')
    plt.savefig('fwaccuracy.png')
    
    plt.clf()
    plt.plot(x, perplexity, label='perplexity')
    plt.savefig('fwperplexity.png')




if __name__ == "__main__":        
    # hyperparameters
    device = torch.device("cuda", index=0)
    batch_size = 240
    epcohs = 10
    seq_len = 7
    lr = 0.00001
   
    # Data creation
    global_embedding, vocab = load_glove(device)

    
    train_data = FWDataset("Dataset/LMTokenizedData/train.json", seq_len, vocab)
    test_data = FWDataset("Dataset/LMTokenizedData/test.json", seq_len, vocab)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Model creation
    Model = ForwardLanguageModel(len(vocab), global_embedding).to(device)
    
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
        
    
    # Saving the model
    torch.save(Model.state_dict(), 'fwmodel.pt')
