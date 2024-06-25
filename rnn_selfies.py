import torch.nn.functional as tf
from torch import nn
from rnn_config import *
import torch
import selfies as sf

class RNNSelfies(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        


    def forward(self, x, pad_mask=None):
        ''' x = batch * seq_len
            pad_mask = batch*seq_len
        '''
        
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = self.linear(x)
        
        return x

    def getDevice(self):
        ''' return device of model
        '''
        device = next(self.parameters()).device

        if device.index is None:
            return device.type
        else:
            return device.type + ":" + str(device.index) 
    
    def generateSmiles(self, batch_size, vocab, max_len=100):
        x = torch.full((batch_size, 1), vocab[START_TOKEN]).to(self.getDevice())
        
        for i in range(1, max_len+1):
            out = self.forward(x)[:, -1]
            out = tf.softmax(out, dim=1)
            out = torch.multinomial(out, 1)
            x = torch.cat((x, out), dim=1)
        
        x = x.detach().cpu().tolist()
        # converting idx to selfies
        results = []
        for i in range(batch_size):
            sentance = vocab.lookup_tokens(x[i])
            
            new_sentance = [] # removing special chars
            for i in range(1, len(sentance)):
                e = sentance[i]
                if e==END_TOKEN: break
                if e==START_TOKEN or e==PAD_TOKEN or e==UNK_TOKEN:
                    new_sentance = ['invalid']
                    break
                
                new_sentance.append(e)
            
            results.append(sf.decoder(''.join(new_sentance)))
        
        return results
    
