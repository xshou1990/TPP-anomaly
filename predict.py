import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from CustomRMTPP import JJJ_RMTPP as RMTPP
from data_utils import collate_fn, AmazonDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Predict with RMTPP model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the test data')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to save predictions')
    parser.add_argument('--num_event_types', type=int, required=True,
                      help='Number of event types (including padding)')
    parser.add_argument('--embed_dim', type=int, required=True,
                      help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, required=True,
                      help='Hidden dimension')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for prediction')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of layers in the model')
    parser.add_argument('--min_sequence_length', type=int, default=2,
                      help='Minimum sequence length to process')
    return parser.parse_args()

def load_model(model_path, num_event_types, embed_dim, hidden_dim, num_layers, device):
    """Load trained model from checkpoint"""
    model = RMTPP(
        num_event_types=num_event_types,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers  # Assuming default 2 layers, can be adjusted
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_and_filter_sequences(data_path, min_length=2):
    """Load sequences with strict filtering and validation"""
    data = pd.read_pickle(data_path)
    
    if 'test' not in data:
        raise ValueError("Test data not found in pickle file")
    
    test_seqs = data['test']
    sequences = []
    
    # Handle both list and dict formats
    if isinstance(test_seqs, dict):
        seq_data = list(test_seqs.values())
    else:
        seq_data = test_seqs
        
    malformed_count = 0

    for seq in seq_data:
        try:
            # Skip empty sequences
            if not seq or len(seq) < min_length:
                continue
                
            # Extract times and types
            times = []
            types = []
            for item in seq:
                try:
                    times.append(float(item['time_since_last_event']))
                    types.append(int(item['type_event']))
                except (KeyError, TypeError, ValueError):
                    continue
                    
            # Verify we have enough valid events
            if len(times) >= min_length and len(types) >= min_length:
                sequences.append((np.array(times), np.array(types)))
                
        except Exception as e:
            malformed_count += 1

            if malformed_count < 10:  # Limit error messages
                continue
            else:
                print(f"Skipping malformed sequence: {str(e)}")
                continue
            
    if not sequences:
        raise ValueError(f"No valid sequences found (minimum length: {min_length})")
        
    print(f"Loaded {len(sequences)} sequences after filtering")

    if malformed_count > 0:
        print(f"Skipped {malformed_count} malformed sequences")

    return sequences

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(
        args.model_path,
        args.num_event_types,
        args.embed_dim,
        args.hidden_dim,
        args.num_layers,
        device
    )

    # Load and filter test sequences
    test_sequences = load_and_filter_sequences(
        args.data_path,
        min_length=args.min_sequence_length
    )

    # Run predictions
    try:
        all_predictions = model.predict_sequences(test_sequences)
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        # Try with individual sequences as fallback
        print("Attempting fallback prediction method...")
        all_predictions = []

        malformed_count = 0

        for seq in test_sequences:
            try:
                preds = model.predict_sequences([seq])
                all_predictions.append(preds)
            except Exception as seq_e:
                malformed_count += 1
                if malformed_count > 10:  # Limit error messages
                    continue
                else:
                    print(f"Skipping sequence due to error: {str(seq_e)}")
                
        if not all_predictions:
            raise RuntimeError("All predictions failed")
        
        print(f"Successfully processed {len(all_predictions)} sequences with fallback method")
        print(f"Skipped {malformed_count} sequences due to errors")
        all_predictions = pd.concat(all_predictions)

    # Save results
    print(all_predictions.head())
    all_predictions.to_csv(args.output_path, index=False)
    print(f"Successfully saved predictions to {args.output_path}")
    print(f"Total predictions: {len(all_predictions)}")

if __name__ == "__main__":
    main()