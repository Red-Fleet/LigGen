import utils as utils
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset, random_split
import selfies as sf
import torch
from rnn_config import *
import utils


class SelfiesDataset(Dataset):
    def __init__(self, smiles_path: list, vocab, max_len=None): # start and end tokens are added
        '''
        if max_len is None: take max_len from dataset
        if max_len == 'avg': take average length from dataset
        '''
        self.vocab = vocab

        # reading smiles from file
        tokenized_sen = []

        with open(smiles_path) as f:
            for l in f:
                selfs = utils.smilesToSelfies(l)
                if selfs is not None:
                    tokenized_sen.append(list(sf.split_selfies(selfs)))

        
        if max_len is None: max_len = max([len(sen) for sen in tokenized_sen])
        if max_len == 'avg': max_len = int(sum([len(sen) for sen in tokenized_sen])/len(tokenized_sen))

        # stripping
        tokenized_sen = [sen[: max_len-2] for sen in tokenized_sen]

        # adding start and end tokens
        tokenized_sen = [[START_TOKEN] + sen + [END_TOKEN] for sen in tokenized_sen]

        # padding
        tokenized_sen = [sen + [PAD_TOKEN]*(max_len-len(sen)) for sen in tokenized_sen]

        # addention mask false at <pad> tokem, true at non pad token
        self.pad_masks = torch.tensor([[PAD_TOKEN==tok for tok in sen] for sen in tokenized_sen])

        # converting to index
        self.data = torch.tensor([self.vocab(sen) for sen in tokenized_sen], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'idx': self.data[idx], 'pad_mask': self.pad_masks[idx]}

