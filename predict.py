import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from JJJRMTPP import JJJ_RMTPP
from train import TaxiDataset, collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions using RMTPP model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--data_path', type=str,
                      default='./Datasets/taxi/',
                      help='Path to data directory')
    parser.add_argument('--batch_size', type=int,
                      default=256,
                      help='Batch size for prediction')
    parser.add_argument('--num_sample', type=int,
                      default=1,
                      help='Number of samples per prediction')
    parser.add_argument('--look_ahead_time', type=float,
                      default=10.0,
                      help='Time window for prediction')
    parser.add_argument('--save_predictions', type=str, required=True,
                      help='Path to save predictions')
    return parser.parse_args()

def load_model(model_path, device):
    """Load trained model with architecture parameters"""
    data_dict = pd.read_pickle(os.path.join(os.path.dirname(model_path), '../Datasets/taxi/train.pkl'))
    num_event_types = data_dict['dim_process']
    
    # Initialize model with default config that matches training
    model = JJJ_RMTPP(
        num_event_types=num_event_types,
        embed_dim=32,
        hidden_dim=16,
        time_embed_size=16,
        num_layers=2,
        num_heads=2,
        use_padding=False
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(model, dataloader, device, num_sample=1, look_ahead=10.0):
    """Make predictions on new sequences"""
    all_sequences = []
    all_pred_types = []
    all_pred_times = []
    
    with torch.no_grad():
        for batch in dataloader:
            times, dts, types, masks = batch
            times, dts, types, masks = (
                times.to(device),
                dts.to(device),
                types.to(device),
                masks.to(device))
            
            # Get predictions for next event
            pred_types, pred_times = model.predict_next_event(dts, types)
            
            # Store results
            seq_lens = masks.sum(1).long()
            for i in range(len(seq_lens)):
                seq_len = seq_lens[i].item()
                if seq_len > 0:
                    all_sequences.append({
                        'input_sequence': types[i, :seq_len].cpu().numpy(),
                        'input_times': times[i, :seq_len].cpu().numpy(),
                        'pred_next_type': pred_types[i].item(),
                        'pred_next_time': pred_times[i].item()
                    })
                    all_pred_types.append(pred_types[i].item())
                    all_pred_times.append(pred_times[i].item())
    
    return {
        'sequences': all_sequences,
        'pred_types': np.array(all_pred_types),
        'pred_times': np.array(all_pred_times)
    }

def main():
    args = parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, device)
    
    # Load data
    print("Loading data for prediction...")
    predict_dict = pd.read_pickle(os.path.join(args.data_path, 'test.pkl'))
    predict_sequences = predict_dict['test']
    print(f"Number of sequences to predict: {len(predict_sequences)}")
    
    # Create dataloader
    predict_dataset = TaxiDataset(predict_sequences)
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Make predictions
    print("Making predictions...")
    predictions = predict(
        model, predict_loader, device,
        num_sample=args.num_sample,
        look_ahead=args.look_ahead_time
    )
    
    # Save predictions
    print(f"Saving predictions to {args.save_predictions}...")
    os.makedirs(os.path.dirname(args.save_predictions), exist_ok=True)
    with open(args.save_predictions, 'wb') as f:
        pickle.dump(predictions, f)
    
    # Print summary
    print("\nPrediction Summary:")
    print("------------------")
    print(f"Number of sequences processed: {len(predictions['sequences'])}")
    print(f"Example predictions saved to: {args.save_predictions}")
    print("------------------")

if __name__ == '__main__':
    main()