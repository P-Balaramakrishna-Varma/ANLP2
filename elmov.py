from LM import *



class Elmov(nn.Module):
    def __init__(self, forward_file, backward_file, len_vocab, gobal_embedding, w1=None):
        super().__init__()
        
        # loading the forward lm
        self.ForwardModel = ForwardLanguageModel(len_vocab, gobal_embedding)
        self.ForwardModel.load_state_dict(torch.load(forward_file))
        self.ForwardModel.requires_grad_(False)
        
        # loaidng the backward lm
        self.BackwardModel = ForwardLanguageModel(len_vocab, gobal_embedding)
        self.BackwardModel.load_state_dict(torch.load(backward_file))
        self.BackwardModel.requires_grad_(False)
        
        # setting the weights
        if(w1 == None):
            self.w1 = nn.Parameter(data=torch.rand(1), requires_grad=True)
        else:
            self.w1 = nn.Parameter(data=torch.tensor([w1]), requires_grad=False)
            
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # forward pass
        ## getting weights
        w1 = self.tanh(self.relu(self.w1))
        
        ## forward rep
        f = self.ForwardModel.forward2(x, w1)
        
        ## backward rep
        b = self.BackwardModel.forward2(x, w1)
        b = torch.flip(b, [1])
        
        ## comined rep (conatination)
        h = torch.cat((f, b), 2)
        return h
    
    
if __name__ == "__main__":
    emd, vocab = load_glove("cuda:0")
    a = Elmov("fwmodel.pt", "bwmodel.pt", len(vocab), emd).to("cuda:0")
    print(a(torch.tensor([[0], [1]]).to("cuda:0")).shape)