from CustomRMTPP import JJJ_RMTPP as RMTPP  # Import your model class
import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset

def load_model(model_path, num_event_types, embed_dim, hidden_dim, device):
    """Load trained model from checkpoint"""
    model = RMTPP(
        num_event_types=num_event_types,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Optimized collate function
def collate_fn(batch):
    times, dts, types = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in times])
    max_len = lengths.max().item()

    padded_times = torch.zeros(len(batch), max_len)
    padded_dts = torch.zeros(len(batch), max_len)
    padded_types = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len)

    for i, (t_seq, dt_seq, t_type) in enumerate(zip(times, dts, types)):
        seq_len = len(t_seq)
        padded_times[i, :seq_len] = t_seq
        padded_dts[i, :seq_len] = dt_seq
        padded_types[i, :seq_len] = t_type
        mask[i, :seq_len] = 1

    return padded_times, padded_dts, padded_types, mask

def df_to_sequences(data_dict, key):
    """
    Converts a dictionary containing sequence data into a list of (times, types) tuples.

    Args:
        data_dict (dict): A dictionary containing the sequence data.
        key (str): The key in the dictionary that holds the list of sequences.

    Returns:
        list: A list of tuples, where each tuple is (times, types) for a sequence.
    """
    sequences = []
    for seq_data in data_dict[key]:
        if not seq_data:  # Skip empty sequences
            continue
        times = np.array([item['time_since_last_event'] 
                        for item in seq_data if item])  # Skip None items
        types = np.array([item['type_event'] 
                        for item in seq_data if item])
        if len(times) > 0 and len(types) > 0:
            sequences.append((times, types))
    return sequences

def load_test_sequences(data_path):
    data = pd.read_pickle(data_path)
    
    if 'test' not in data:
        raise ValueError("Test data not found in pickle file")
    
    test_seqs = data['test']
    
    # Handle different data formats
    if isinstance(test_seqs, dict):
        sequences = []
        for seq_data in test_seqs.values():
            if not seq_data:
                continue
            times = np.array([item['time_since_last_event'] for item in seq_data])
            types = np.array([item['type_event'] for item in seq_data])
            sequences.append((times, types))
    elif isinstance(test_seqs, list):
        sequences = []
        for seq_data in test_seqs:
            if not seq_data:
                continue
            times = np.array([item['time_since_last_event'] for item in seq_data])
            types = np.array([item['type_event'] for item in seq_data])
            sequences.append((times, types))
    else:
        raise ValueError(f"Unknown test data format: {type(test_seqs)}")
    
    return sequences

def process_prediction_data(pickle_data):
    """
    Process prediction data from EasyTPP output pickle file into a structured DataFrame.
    
    Args:
        pickle_data (OrderedDict): Loaded data from pred.pkl containing:
            - 'acc': accuracy score
            - 'rmse': RMSE score
            - 'pred': list of predicted sequences
            - 'label': list of true label sequences
    
    Returns:
        tuple: (pd.DataFrame with prediction results, dict with metrics)
    """
    # Initialize lists to hold flattened data
    rows = []

    # Extract metrics
    metrics = {
        'accuracy': pickle_data['acc'],
        'rmse': pickle_data['rmse']
    }

    # Iterate through each sequence in predictions and labels
    for seq_idx, (pred_seq, label_seq) in enumerate(zip(pickle_data['pred'], pickle_data['label'])):
        # Convert to numpy arrays if they aren't already
        pred_seq = np.array(pred_seq)
        label_seq = np.array(label_seq)
        
        # Filter out padding (event_type = 16)
        valid_pred_mask = pred_seq[:, 0] != 16.0
        valid_label_mask = label_seq[:, 0] != 16.0
        
        # Get only valid events (non-padding)
        pred_events = pred_seq[valid_pred_mask]
        label_events = label_seq[valid_label_mask]
        
        # Initialize cumulative time for this sequence
        pred_time_since_start = 0.0
        true_time_since_start = 0.0

        # Pair predictions with labels
        for event_idx, (pred, label) in enumerate(zip(pred_events, label_events)):
            # Update cumulative times
            pred_time_since_start += pred[1]
            true_time_since_start += label[1]
            
            rows.append({
                'sequence_id': seq_idx + 1,
                'event_idx': event_idx + 1,
                'pred_event_type': int(pred[0]),
                'pred_time_since_last': float(pred[1]),
                'pred_time_since_start': float(pred_time_since_start),
                'true_event_type': int(label[0]),
                'true_time_since_last': float(label[1]),
                'true_time_since_start': float(true_time_since_start),
            })

    # Create DataFrame
    results_df = pd.DataFrame(rows)
    
    return results_df, metrics


def save_results(results_df, metrics, results_dir, model_id='RMTPP'):
    """
    Save processed results to files.
    
    Args:
        results_df (pd.DataFrame): Processed results dataframe
        metrics (dict): Dictionary containing evaluation metrics
        results_dir (str): Directory to save output files
    """
    # Save results to CSV
    results_file = os.path.join(results_dir, f"{model_id}_prediction_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Prediction results saved to: {results_file}")
    
    # Save metrics to text file
    metrics_file = os.path.join(results_dir, f"{model_id}_evaluation_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"RMSE: {metrics['rmse']:.4f}\n")
    print(f"Evaluation metrics saved to: {metrics_file}")