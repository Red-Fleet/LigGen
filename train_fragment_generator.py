import torch
import argparse
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rnn_selfies import RNNSelfies
from rnn_config import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_canonical_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None
from torch.utils.data import DataLoader
from selfies_dataset import SelfiesDataset
from torch import nn
import torch.optim as optim
from tqdm import tqdm

# Training function
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    count = 200
    pbar = tqdm(total=len(train_loader))
    
    for i, e in enumerate(train_loader):
        x = e['idx'].to(device)
        pad_mask = e['pad_mask'].to(device)

        x_input = x[:, :-1]
        pad_mask = pad_mask[:, :-1]
        y_expected = x[:, 1:]
        output = model(x_input, pad_mask)
        #print(output.shape, y_expected.shape)

        output = torch.flatten(output, start_dim=0, end_dim=1)
        y_expected = torch.flatten(y_expected, start_dim=0, end_dim=1)
        loss = criterion(output, y_expected)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

        pbar.update(1)
        #if (count+1)%100 == 0: print(loss.item())
    
    pbar.close()
    total_loss = total_loss/len(train_loader)
    
    return total_loss

# valid function
def valid(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for e in val_loader:
            x = e['idx'].to(device)
            pad_mask = e['pad_mask'].to(device)

            x_input = x[:, :-1]
            pad_mask = pad_mask[:, :-1]
            y_expected = x[:, 1:]

            output = model(x_input, pad_mask)
            
            output = torch.flatten(output, start_dim=0, end_dim=1)
            y_expected = torch.flatten(y_expected, start_dim=0, end_dim=1)
            loss = criterion(output, y_expected)

            total_loss += loss.item()
        
    total_loss = total_loss/len(val_loader)
   
    return total_loss
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Train-fragment-generator',
                        description='Train fragment generation model',
                        )

    default_out_model_params = 'model.pt'
    default_max_len = 100
    default_batch_size = 512
    default_epoch = 1
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser.add_argument('-i', '--input_smiles', type=str, required=True,
                        help='path of file containing fragments in smile format')
    
    parser.add_argument('-ip', '--in_model_params', type=str, default=None,
                        help=f'path for initializing model params (default= random)')
    
    parser.add_argument('-op', '--out_model_params', type=str, default=default_out_model_params,
                        help=f'path for saving model params (default= {default_out_model_params})')
    
    parser.add_argument('-b', '--batch_size', type=int, default=default_batch_size,
                        help=f'batch size (default= {default_batch_size})')
    
    parser.add_argument('-e', '--epoch', type=int, default=default_epoch,
                        help=f'number of epoches (default= {default_epoch})')
    
    parser.add_argument('-l', '--max_len', type=int, default=default_max_len,
                        help=f'length of input tokens(selfies tokens) (default= {default_max_len})')
    
    parser.add_argument('-d', '--device', type=str, default=default_device,
                        help=f'length of input tokens(selfies tokens) (default= {default_device})')
    
    parser.add_argument('-f', type=int, default=0,
                        help='Epoch frequency for generating and calculating similarity (default=0, no calc)')
    
    parser.add_argument('-y', type=int, default=-1,
                        help='Number of smiles to generate (default=10*batch_size)')
    
    parser.add_argument('-sc', '--similarity_cutoff', type=float, default=0.5,
                        help='Threshold for stopping training (default=0.5)')
    
    args = parser.parse_args()
    
    if args.y == -1:
        args.y = 10 * args.batch_size

    for k, v in args.__dict__.items():
        print(k, ":", v)

    vocab = get_vocab()
    model = RNNSelfies(vocab_size=len(vocab), 
        embed_dim=256,
        hidden_size=512,
        num_layers=3,
        dropout=0)

    if args.in_model_params is not None: model.load_state_dict(torch.load(args.in_model_params, map_location=args.device))

    model = model.to(args.device)

    dataset = SelfiesDataset(args.input_smiles, vocab=vocab, max_len=args.max_len)
    train_loader = DataLoader(dataset, args.batch_size)

    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=vocab[PAD_TOKEN])
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True, weight_decay=0.0001)

    # learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=0.5, patience=4,
        cooldown=8, min_lr=0.0000001
    )

    if args.f > 0:
        print("Pre-calculating training set unique smiles (x_set)...")
        x_set = set()
        with open(args.input_smiles, 'r') as f_in:
            for line in f_in:
                if not line.strip(): continue
                smi = line.strip().split(',')[0].split()[0]
                can = get_canonical_smiles(smi)
                if can:
                    x_set.add(can)
        print(f"Loaded {len(x_set)} unique canonical smiles from training set.")

    for i in range(args.epoch):
        print('\n')
        print(f'Epoch : {i+1}')
        loss = train(model, optimizer, criterion, train_loader, args.device)
        print(f'Loss : {loss}')
        scheduler.step(loss)

        if args.f > 0 and (i + 1) % args.f == 0:
            print(f"Generating ~{args.y} smiles to check similarity...")
            gen_smiles = []
            num_batches = max(1, args.y // args.batch_size)
            model.eval()
            with torch.no_grad():
                for _ in range(num_batches):
                    gen_smiles.extend(model.generateSmiles(args.batch_size, vocab, max_len=args.max_len))
            
            y_set = set()
            for smi in gen_smiles:
                if smi and smi != 'invalid':
                    can = get_canonical_smiles(smi)
                    if can:
                        y_set.add(can)
            
            if len(y_set) > 0:
                overlap_ratio = len(x_set.intersection(y_set)) / len(x_set.union(y_set))
                print(f"Jaccard Similarity: {overlap_ratio:.2%} ({len(x_set.intersection(y_set))} overlapping / {len(x_set.union(y_set))} total unique)")
                if overlap_ratio >= args.similarity_cutoff:
                    print(f"Stopping training: Similarity >= {args.similarity_cutoff}")
                    break
            else:
                print("Could not generate any valid unique smiles.")

    
    torch.save(model.state_dict(), args.out_model_params)
        
        