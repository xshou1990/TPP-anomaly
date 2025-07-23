import os
import glob
import yaml
import pickle
import data_utils
import pandas as pd
import numpy as np
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

# ====== 1. Define Paths ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Datasets", "amazon")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parameters for the FullyNN model
#  Data parameters
data_id = "amazon"
data_format = "pkl"
train_dir = os.path.join(DATASET_DIR, "train.pkl")
valid_dir = os.path.join(DATASET_DIR, "dev.pkl")
test_dir = os.path.join(DATASET_DIR, "test.pkl")
data_specs = {
    "num_event_types": 16,
    "pad_token_id": 16,
    "padding_side": "right"
}

#  Configuration parameters
model_id = "FullyNN"
hidden_size = 32
time_embed_size = 16
num_layers = 2
batch_size = 256
epochs = 10
valid_freq = 1
seed = 2019
dropout = 0.0
learning_rate = 1e-3

rnn_type = "LSTM"  
use_ln = False
num_heads = 2
num_dense_layers = 2  
dtime_max = 5.0      
num_mlp_layers = 3
proper_marked_intensities = True
#  Thinning parameters (if needed)
num_sequence = 10
num_sample = 1
num_exp = 500
look_ahead_time = 10
patience_counter = 5
over_sample_rate = 5
num_samples_boundary = 5
intensity_lower_bound = 1e-6
intensity_higher_bound = 100

# ====== 2. Training Configuration ======
train_config = {
    "pipeline_config_id": "runner_config",
    "data": {
        data_id: {
            "data_format": data_format,
            "train_dir": train_dir,
            "valid_dir": valid_dir,
            "test_dir": test_dir,
            "data_specs": data_specs
        }
    },
    f"{model_id}_train": {
        "base_config": {
            "stage": "train",
            "backend": "torch",
            "dataset_id": data_id,
            "runner_id": "std_tpp",
            "model_id": model_id,
            "base_dir": CHECKPOINT_DIR
        },
        "trainer_config": {
            "batch_size": batch_size,
            "max_epoch": epochs,
            "shuffle": False,
            "optimizer": "adam",
            "learning_rate": learning_rate,
            "valid_freq": valid_freq,
            "use_tfb": False,
            "metrics": ["acc"],
            "seed": seed,
            "gpu": 0
        },
        "model_config": {
            "rnn_type": rnn_type,
            "hidden_size": hidden_size,
            "time_embed_size": time_embed_size,
            "num_layers": num_layers,
            "use_ln": use_ln,
            "dropout": dropout,
            "num_heads": num_heads,
            "seed": seed,
            "model_specs": {
                "num_mlp_layers": num_mlp_layers,
                "proper_marked_intensities": proper_marked_intensities,
            }
        }
    }
}

# Save training config
train_config_path = os.path.join(BASE_DIR, f"{model_id}_train_config.yaml")
with open(train_config_path, "w") as f:
    yaml.dump(train_config, f)

# ====== 3. Run Training ======
print("Starting training...")
train_config = Config.build_from_yaml_file(train_config_path, experiment_id=f'{model_id}_train')
model_runner = Runner.build_from_config(train_config)
model_runner.run()

# ====== 4. Find Latest Model ======
def find_latest_saved_model(checkpoint_dir):
    """Finds the most recent 'saved_model' in run folders"""
    run_folders = glob.glob(os.path.join(checkpoint_dir, "*_*_*"))
    
    if not run_folders:
        raise FileNotFoundError(f"No run folders found in: {checkpoint_dir}")
    
    latest_run_folder = max(run_folders, key=os.path.getmtime)
    saved_model_path = os.path.join(latest_run_folder, "models", "saved_model")
    
    if not os.path.exists(saved_model_path):
        raise FileNotFoundError(f"No 'saved_model' found in: {latest_run_folder}/models/")
    
    return saved_model_path

try:
    pretrained_model_path = find_latest_saved_model(CHECKPOINT_DIR)
    print(f"Latest model found: {pretrained_model_path}")

    # ====== 5. Evaluation Configuration ======
    eval_config = {
        "pipeline_config_id": "runner_config",
        "data": {
            data_id: {
                "data_format": data_format,
                "train_dir": train_dir,
                "valid_dir": valid_dir,
                "test_dir": test_dir,
                "data_specs": data_specs
            }
        },
        f"{model_id}_eval": {
            "base_config": {
                "stage": "eval",
                "backend": "torch",
                "dataset_id": data_id,
                "runner_id": "std_tpp",
                "base_dir": os.path.dirname(os.path.dirname(pretrained_model_path)),
                "model_id": model_id,
                "pred_dir": BASE_DIR,
                "out_config_dir": os.path.join(RESULTS_DIR, f'{model_id}_test_output.yaml')
            },
            "trainer_config": {
                "batch_size": batch_size,
                "max_epoch": epochs,
                "shuffle": False,
                "optimizer": "adam",
                "learning_rate": learning_rate,
                "valid_freq": valid_freq,
                "use_tfb": False,
                "metrics": ["acc"],
                "seed": seed,
                "gpu": 0
            },
            "model_config": {
                "rnn_type": rnn_type,
                "hidden_size": hidden_size,
                "time_embed_size": time_embed_size,
                "num_layers": num_layers,
                "use_ln": use_ln,
                "dropout": dropout,
                "num_heads": num_heads,
                "seed": seed,
                "model_specs": {
                    "num_mlp_layers": num_mlp_layers,
                    "proper_marked_intensities": proper_marked_intensities,
                    }
            }
        }
    }

    # Save eval config
    eval_config_path = os.path.join(BASE_DIR, f"{model_id}_eval_config.yaml")
    with open(eval_config_path, "w") as f:
        yaml.dump(eval_config, f)

    # ====== 6. Run Evaluation ======
    print("Starting evaluation...")
    eval_config = Config.build_from_yaml_file(eval_config_path, experiment_id=f'{model_id}_eval')
    model_runner = Runner.build_from_config(eval_config)
    model_runner.run()  # Capture evaluation results

    # ====== 7. Process Prediction Results ======
    print("Processing prediction results...")
    pred_file = os.path.join(BASE_DIR, "pred.pkl")  # Look in current directory

    if os.path.exists(pred_file):
        with open(pred_file, 'rb') as f:
            data = pickle.load(f)

        # Process the data
        results_df, metrics = data_utils.process_prediction_data(data)

        # Save results (assuming you have a RESULTS_DIR defined)
        data_utils.save_results(results_df, metrics, RESULTS_DIR, model_id=model_id)

        # You can also access the metrics directly
        print(f"Model Accuracy: {metrics['accuracy']:.4f}")
        print(f"Model RMSE: {metrics['rmse']:.4f}")
    else:
        print(f"Warning: Prediction file not found at {pred_file}")

except Exception as e:
    print(f"Error: {e}")
    print("\nDebugging Tips:")
    print(f"1. Does the checkpoint directory exist? Check: {CHECKPOINT_DIR}")
    print("2. Are there run folders inside it?")
    print("3. Does each run folder contain a 'models/saved_model' file?")