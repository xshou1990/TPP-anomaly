import os
import argparse
import time
import pickle
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from JJJRMTPP import JJJ_RMTPP

def parse_args():
    parser = argparse.ArgumentParser(description='Train RMTPP model on Taxi dataset')
    parser.add_argument('--embed_dim', type=int, 
                       default=int(os.environ.get('MODEL_EMBED_DIM', 32)),
                       help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int,
                       default=int(os.environ.get('MODEL_HIDDEN_DIM', 16)),
                       help='Hidden dimension')
    parser.add_argument('--batch_size', type=int,
                       default=int(os.environ.get('BATCH_SIZE', 256)),
                       help='Batch size')
    parser.add_argument('--num_layers', type=int,
                       default=int(os.environ.get('MODEL_NUM_LAYERS', 2)),
                       help='Number of RNN layers')
    parser.add_argument('--epochs', type=int,
                       default=int(os.environ.get('EPOCHS', 200)),
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float,
                       default=float(os.environ.get('LEARNING_RATE', 0.001)),
                       help='Learning rate')
    parser.add_argument('--time_embed_size', type=int,
                       default=int(os.environ.get('MODEL_TIME_EMBED_SIZE', 16)),
                       help='Time embedding dimension')
    parser.add_argument('--num_heads', type=int,
                       default=int(os.environ.get('MODEL_NUM_HEADS', 2)),
                       help='Number of attention heads')
    parser.add_argument('--mc_samples', type=int,
                       default=int(os.environ.get('MODEL_MC_SAMPLES', 20)),
                       help='MC samples for integral approximation')
    parser.add_argument('--integral_samples', type=int,
                       default=int(os.environ.get('MODEL_INTEGRAL_SAMPLES', 20)),
                       help='Samples for loss integral')
    parser.add_argument('--data_path', type=str,
                       default=os.environ.get('DATA_PATH', './Datasets/taxi/'),
                       help='Path to data directory')
    parser.add_argument('--save_dir', type=str,
                       default=os.environ.get('MODEL_DIR', './models'),
                       help='Directory to save models')
    parser.add_argument('--seed', type=int,
                       default=int(os.environ.get('SEED', 2019)),
                       help='Random seed')
    return parser.parse_args()

class TaxiDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def collate_fn(batch_sequences):
    """Convert list of sequences to padded tensors"""
    max_len = max(len(seq) for seq in batch_sequences)
    batch_size = len(batch_sequences)
    
    times = torch.zeros(batch_size, max_len)
    dts = torch.zeros(batch_size, max_len)
    types = torch.zeros(batch_size, max_len, dtype=torch.long)
    masks = torch.zeros(batch_size, max_len)
    
    for i, seq in enumerate(batch_sequences):
        seq_len = len(seq)
        seq_times = [event['time_since_start'] for event in seq]
        seq_types = [event['type_event'] for event in seq]
        
        times[i, :seq_len] = torch.tensor(seq_times)
        types[i, :seq_len] = torch.tensor(seq_types)
        masks[i, :seq_len] = 1.0
        
        if seq_len > 1:
            dts[i, 1:seq_len] = times[i, 1:seq_len] - times[i, :seq_len-1]
            
    return times, dts, types, masks

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_events = 0
    
    for batch in dataloader:
        times, dts, types, masks = batch
        times, dts, types, masks = (
            times.to(device),
            dts.to(device),
            types.to(device),
            masks.to(device))
        
        optimizer.zero_grad()
        loss = model.compute_loss(dts, types, masks)
        loss.backward()
        optimizer.step()
        
        batch_events = masks.sum().item()
        total_loss += loss.item() * batch_events
        total_events += batch_events
    
    return total_loss / max(total_events, 1)
    
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_events = 0
    
    with torch.no_grad():
        for batch in dataloader:
            times, dts, types, masks = batch
            times, dts, types, masks = (
                times.to(device),
                dts.to(device),
                types.to(device),
                masks.to(device))
            
            loss = model.compute_loss(dts, types, masks)
            batch_events = masks.sum().item()
            total_loss += loss.item() * batch_events
            total_events += batch_events
    
    return total_loss / max(total_events, 1)

def main():
    args = parse_args()
    
    # Print configuration
    print("\nTraining Configuration:")
    print("----------------------")
    for arg in vars(args):
        print(f"{arg:>20}: {getattr(args, arg)}")
    print("----------------------\n")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_dict = pd.read_pickle(os.path.join(args.data_path, 'train.pkl'))
    valid_dict = pd.read_pickle(os.path.join(args.data_path, 'dev.pkl'))
    
    train_sequences = train_dict['train']
    valid_sequences = valid_dict['dev']
    num_event_types = train_dict['dim_process']
    
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(valid_sequences)}")
    print(f"Number of event types: {num_event_types}")
    
    # Create datasets and dataloaders
    train_dataset = TaxiDataset(train_sequences)
    valid_dataset = TaxiDataset(valid_sequences)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = JJJ_RMTPP(
        num_event_types=num_event_types,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        time_embed_size=args.time_embed_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mc_num_sample_per_step=args.mc_samples,
        loss_integral_num_sample_per_step=args.integral_samples,
        use_padding=False
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    no_improve = 0
    patience = int(os.environ.get('PATIENCE_COUNTER', 5))
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train and validate
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, valid_loader, device)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            model_path = os.path.join(args.save_dir, os.environ.get('MODEL_FILE', 'rmtpp_taxi_best.pt'))
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model to {model_path}")
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Print stats
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch:03d} | Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"No Improve: {no_improve}/{patience}")
    
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()