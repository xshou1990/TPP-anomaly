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
    parser = argparse.ArgumentParser(description='Test RMTPP model on Taxi dataset')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, 
                       default=os.environ.get('DATA_PATH', './Datasets/taxi/'), 
                       help='Path to data directory')
    parser.add_argument('--batch_size', type=int, 
                       default=int(os.environ.get('BATCH_SIZE', 256)), 
                       help='Batch size for evaluation')
    parser.add_argument('--save_results', type=str, 
                       default=os.path.join(os.environ.get('RESULTS_DIR', './results'), 
                                          os.environ.get('RESULTS_FILE', 'taxi_predictions.pkl')),
                       help='Path to save predictions')
    return parser.parse_args()

def load_model(model_path, device):
    """Load trained model with architecture parameters from config.sh"""
    data_dict = pd.read_pickle(os.path.join(os.path.dirname(model_path), '../Datasets/taxi/test.pkl'))
    num_event_types = data_dict['dim_process']
    
    # Get parameters from environment (set by config.sh)
    model_config = {
        'embed_dim': int(os.environ.get('MODEL_EMBED_DIM', 32)),
        'hidden_dim': int(os.environ.get('MODEL_HIDDEN_DIM', 16)),
        'time_embed_size': int(os.environ.get('MODEL_TIME_EMBED_SIZE', 16)),
        'num_layers': int(os.environ.get('MODEL_NUM_LAYERS', 2)),
        'num_heads': int(os.environ.get('MODEL_NUM_HEADS', 2)),
        'mc_samples': int(os.environ.get('MODEL_MC_SAMPLES', 20))
    }
    
    print("\nLoading model with configuration:")
    for k, v in model_config.items():
        print(f"{k:>15}: {v}")
    
    # Initialize model
    model = JJJ_RMTPP(
        num_event_types=num_event_types,
        embed_dim=model_config['embed_dim'],
        hidden_dim=model_config['hidden_dim'],
        time_embed_size=model_config['time_embed_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        mc_num_sample_per_step=model_config['mc_samples'],
        use_padding=False
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate(model, dataloader, device):
    """Evaluate model on test set and return predictions"""
    all_pred_types = []
    all_true_types = []
    all_pred_times = []
    all_true_times = []
    all_sequences = []
    
    with torch.no_grad():
        for batch in dataloader:
            times, dts, types, masks = batch
            times, dts, types, masks = (
                times.to(device),
                dts.to(device),
                types.to(device),
                masks.to(device))
            
            # Get predictions for next event
            pred_types, pred_times = model.predict_next_event(dts[:, :-1], types[:, :-1])
            
            # Store results (only for last event in each sequence)
            seq_lens = masks.sum(1).long() - 1
            for i in range(len(seq_lens)):
                if seq_lens[i] > 0:  # Only for sequences with at least 2 events
                    idx = seq_lens[i] - 1
                    all_pred_types.append(pred_types[i].item())
                    all_true_types.append(types[i, idx+1].item())
                    all_pred_times.append(pred_times[i].item())
                    all_true_times.append(times[i, idx+1].item())
                    all_sequences.append({
                        'input_sequence': types[i, :idx+1].cpu().numpy(),
                        'true_next_type': types[i, idx+1].item(),
                        'pred_next_type': pred_types[i].item(),
                        'true_next_time': times[i, idx+1].item(),
                        'pred_next_time': pred_times[i].item()
                    })
    
    return {
        'pred_types': np.array(all_pred_types),
        'true_types': np.array(all_true_types),
        'pred_times': np.array(all_pred_times),
        'true_times': np.array(all_true_times),
        'sequences': all_sequences
    }

def calculate_metrics(results):
    """Calculate evaluation metrics"""
    # Type prediction accuracy
    type_acc = (results['pred_types'] == results['true_types']).mean()
    
    # Time prediction errors
    time_errors = results['pred_times'] - results['true_times']
    
    # MAE (Mean Absolute Error)
    time_mae = np.abs(time_errors).mean()
    
    # RMSE (Root Mean Square Error)
    time_rmse = np.sqrt(np.mean(time_errors**2))
    
    return {
        'type_accuracy': type_acc,
        'time_mae': time_mae,
        'time_rmse': time_rmse,
        'num_samples': len(results['true_types'])
    }

def main():
    args = parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, device)
    
    # Load test data
    print("Loading test data...")
    test_dict = pd.read_pickle(os.path.join(args.data_path, 'test.pkl'))
    test_sequences = test_dict['test']
    print(f"Number of test sequences: {len(test_sequences)}")
    
    # Create dataloader
    test_dataset = TaxiDataset(test_sequences)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Evaluate
    print("Running evaluation...")
    results = evaluate(model, test_loader, device)
    metrics = calculate_metrics(results)
    
    # Save results
    print(f"Saving results to {args.save_results}...")
    os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
    with open(args.save_results, 'wb') as f:
        pickle.dump({
            'results': results,
            'metrics': metrics,
            'config': {
                'batch_size': args.batch_size,
                'model_path': args.model_path,
                **{k: v for k, v in os.environ.items() if k.startswith('MODEL_')}
            }
        }, f)
    
    # Print summary
    print("\nEvaluation Results:")
    print("------------------")
    print(f"Event Type Accuracy: {metrics['type_accuracy']:.4f}")
    print(f"Time Prediction MAE: {metrics['time_mae']:.4f}")
    print(f"Time Prediction RMSE: {metrics['time_rmse']:.4f}")
    print(f"Number of Samples: {metrics['num_samples']}")
    print("------------------")

if __name__ == '__main__':
    main()