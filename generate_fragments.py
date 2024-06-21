import torch
import argparse
from rnn_selfies import RNNSelfies
from rnn_config import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from selfies_dataset import SelfiesDataset
from torch import nn
import torch.optim as optim
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Fragment-generation',
                        description='Program to generate fragments using rnn',
                        epilog="total_frags = iteration*batch_size"
                        )

    default_model_params = 'model.pt'
    default_max_len = 100
    default_batch_size = 64
    default_iteration = 10
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser.add_argument('-p', '--model_params', type=str, default=default_model_params,
                        help=f'path of model parameters (default= {default_model_params})')
    
    parser.add_argument('-o', '--out_path', type=str,required=True,
                        help='path for saving generated smiles')
    
    parser.add_argument('-b', '--batch_size', type=int, default=default_batch_size,
                        help=f'batch size (default= {default_batch_size})')
    
    parser.add_argument('-i', '--iteration', type=int, default=default_iteration,
                        help=f'number of iterations (default= {default_iteration})')
    
    parser.add_argument('-l', '--max_len', type=int, default=default_max_len,
                        help=f'length of input tokens(selfies tokens) (default= {default_max_len})')
    
    parser.add_argument('-d', '--device', type=str, default=default_device,
                        help=f'length of input tokens(selfies tokens) (default= {default_device})')
    
    
    args = parser.parse_args()

    vocab = get_vocab()
    model = RNNSelfies(vocab_size=len(vocab),
        embed_dim=256,
        hidden_size=512,
        num_layers=3,
        dropout=0)

    model.load_state_dict(torch.load(args.model_params))
    model = model.to(args.device)

    smiles_list = []
    
    pbar = tqdm(total=args.iteration*args.batch_size)
    for i in range(args.iteration):
        smiles_list += model.generateSmiles(batch_size=args.batch_size, vocab=vocab, max_len=args.max_len)
        pbar.update(args.batch_size)
    
    pbar.close()
    
    with open(args.out_path, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + '\n')
    
