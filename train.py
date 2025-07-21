import argparse
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from CustomRMTPP import JJJ_RMTPP as RMTPP # Import your RMTPP class
import numpy as np
from data_utils import df_to_sequences, collate_fn, load_model, AmazonDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train RMTPP model')
    parser.add_argument('--num_event_types', type=int, default=None,
                       help='Number of event types (including padding)')
    parser.add_argument('--embed_dim', type=int, default=32,
                       help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of layers in the model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--data_path', type=str, default='./Datasets/amazon/',
                       help='Path to training data')
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='Directory to save models')
    return parser.parse_args()

def main():
    
     # Parse arguments
    args = parse_args()

    # Load data
    train_data_dict = pd.read_pickle(args.data_path + 'train.pkl')
    test_data_dict = pd.read_pickle(args.data_path + 'test.pkl')
    dev_data_dict = pd.read_pickle(args.data_path + 'dev.pkl')

    # Convert to sequences
    train_sequences = df_to_sequences(train_data_dict, 'train')
    test_sequences = df_to_sequences(test_data_dict, 'test')
    dev_sequences = df_to_sequences(dev_data_dict, 'dev')

    # Shift event types to reserve 0 for padding
    for seq in train_sequences + test_sequences + dev_sequences:
        seq[1][:] = seq[1] + 1

    # Calculate number of event types (including padding)
    max_etype = 0
    for seq in train_sequences + test_sequences + dev_sequences:
        if len(seq[1]) > 0:  # Check if the sequence has any event types
            current_max = np.max(seq[1])
            if current_max > max_etype:
                max_etype = current_max

    num_event_types = max_etype + 1
    print(f"Number of event types (including padding): {num_event_types}")

    # Use calculated num_event_types if not provided
    if args.num_event_types is None:
        args.num_event_types = num_event_types
    elif args.num_event_types < num_event_types:
        print(f"Warning: Provided num_event_types ({args.num_event_types}) is smaller than calculated ({num_event_types})")

    # Create datasets
    train_dataset = AmazonDataset(train_sequences)
    val_dataset = AmazonDataset(dev_sequences)  # Using dev as validation
    test_dataset = AmazonDataset(test_sequences)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn
    )

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RMTPP(
        num_event_types=args.num_event_types,  # Now guaranteed to have a value
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss = model.train_epoch(train_loader, optimizer)
        val_loss = model.validate(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 
                      f"{args.save_dir}/model_epoch_{epoch+1}.pt")

    # Final evaluation on test set
    test_loss = model.validate(test_loader)
    print(f"Final Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()