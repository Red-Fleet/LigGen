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

    default_model_params = 'model.pt'
    default_max_len = 100
    default_batch_size = 512
    default_epoch = 1
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser.add_argument('-i', '--input_smiles', type=str, required=True,
                        help='path of file containing fragments in smile format')
    
    parser.add_argument('-p', '--model_params', type=str, default=default_model_params,
                        help=f'path for saving model parameters (default= {default_model_params})')
    
    parser.add_argument('-b', '--batch_size', type=int, default=default_batch_size,
                        help=f'batch size (default= {default_batch_size})')
    
    parser.add_argument('-e', '--epoch', type=int, default=default_epoch,
                        help=f'number of epoches (default= {default_epoch})')
    
    parser.add_argument('-l', '--max_len', type=int, default=default_max_len,
                        help=f'length of input tokens(selfies tokens) (default= {default_max_len})')
    
    parser.add_argument('-d', '--device', type=str, default=default_device,
                        help=f'length of input tokens(selfies tokens) (default= {default_device})')
    
    args = parser.parse_args()

    for k, v in args.__dict__.items():
        print(k, ":", v)

    model = RNNSelfies(vocab_size=len(VOCAB), 
        embed_dim=256,
        hidden_size=512,
        num_layers=3,
        dropout=0)

    model = model.to(args.device)

    dataset = SelfiesDataset(args.input_smiles, max_len=args.max_len)
    train_loader = DataLoader(dataset, args.batch_size)

    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=VOCAB[PAD_TOKEN])
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True, weight_decay=0.0001)

    # learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=0.5, patience=5,
        cooldown=10, min_lr=0.0001
    )

    for i in range(args.epoch):
        print('\n')
        print(f'Epoch : {i+1}')
        loss = train(model, optimizer, criterion, train_loader, args.device)
        print(f'Loss : {loss}')

    
    torch.save(model.state_dict(), args.model_params)
        
        