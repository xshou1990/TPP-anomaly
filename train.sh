#!/bin/bash

# Initialize conda
eval "$(conda shell.bash hook)"

# --------------------------
# Load Configuration
# --------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_PATH="$SCRIPT_DIR/config.sh"

if [ -f "$CONFIG_PATH" ]; then
    echo "Loading configuration from $CONFIG_PATH"
    source "$CONFIG_PATH"
else
    echo "Warning: config.sh not found. Using default values."
fi

# Set default values if not defined
export MODEL_EMBED_DIM=${MODEL_EMBED_DIM:-32}
export MODEL_HIDDEN_DIM=${MODEL_HIDDEN_DIM:-16}
export MODEL_TIME_EMBED_SIZE=${MODEL_TIME_EMBED_SIZE:-16}
export MODEL_NUM_LAYERS=${MODEL_NUM_LAYERS:-2}
export MODEL_NUM_HEADS=${MODEL_NUM_HEADS:-2}
export MODEL_MC_SAMPLES=${MODEL_MC_SAMPLES:-20}
export MODEL_INTEGRAL_SAMPLES=${MODEL_INTEGRAL_SAMPLES:-20}
export BATCH_SIZE=${BATCH_SIZE:-256}
export EPOCHS=${EPOCHS:-200}
export LEARNING_RATE=${LEARNING_RATE:-0.001}
export PATIENCE_COUNTER=${PATIENCE_COUNTER:-5}
export DATA_PATH=${DATA_PATH:-"./Datasets/taxi/"}
export MODEL_DIR=${MODEL_DIR:-"./models"}
export MODEL_FILE=${MODEL_FILE:-"rmtpp_taxi_best.pt"}
export SEED=${SEED:-2019}

# --------------------------
# Environment Verification
# --------------------------
echo "Verifying conda environment..."
if ! conda activate tpp_env; then
    echo "Error: Failed to activate 'tpp_env' environment."
    echo "Please make sure:"
    echo "1. You've run setup_env.sh first"
    echo "2. The environment exists (check with: conda env list)"
    exit 1
fi

# Verify Python and packages
echo "Environment activated. Checking packages..."
python -c "
try:
    import torch, pandas, numpy
    print('Required packages verified.')
except ImportError as e:
    print(f'Missing package: {e}')
    exit(1)
"

# --------------------------
# Directory Preparation
# --------------------------
echo "Preparing directories..."
mkdir -p "$MODEL_DIR" || {
    echo "Error: Failed to create model directory $MODEL_DIR"
    exit 1
}

# --------------------------
# Run Training
# --------------------------
echo "Starting training with parameters:"
echo "---------------------------------"
echo "Embed dim:      $MODEL_EMBED_DIM"
echo "Hidden dim:     $MODEL_HIDDEN_DIM"
echo "Time embed:     $MODEL_TIME_EMBED_SIZE"
echo "Num layers:     $MODEL_NUM_LAYERS"
echo "Num heads:      $MODEL_NUM_HEADS"
echo "MC samples:     $MODEL_MC_SAMPLES"
echo "Integral samples: $MODEL_INTEGRAL_SAMPLES"
echo "Batch size:     $BATCH_SIZE"
echo "Epochs:        $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Patience:      $PATIENCE_COUNTER"
echo "Data path:     $DATA_PATH"
echo "Save dir:      $MODEL_DIR"
echo "Model file:    $MODEL_FILE"
echo "Seed:         $SEED"
echo "---------------------------------"

# Run training with all parameters from environment
python train.py \
    --embed_dim "$MODEL_EMBED_DIM" \
    --hidden_dim "$MODEL_HIDDEN_DIM" \
    --batch_size "$BATCH_SIZE" \
    --num_layers "$MODEL_NUM_LAYERS" \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --time_embed_size "$MODEL_TIME_EMBED_SIZE" \
    --num_heads "$MODEL_NUM_HEADS" \
    --mc_samples "$MODEL_MC_SAMPLES" \
    --integral_samples "$MODEL_INTEGRAL_SAMPLES" \
    --data_path "$DATA_PATH" \
    --save_dir "$MODEL_DIR" \
    --seed "$SEED"

# --------------------------
# Completion Handling
# --------------------------
if [ $? -eq 0 ]; then
    echo -e "\nTraining completed successfully!"
    echo "Model saved to: $MODEL_DIR/$MODEL_FILE"
else
    echo -e "\nTraining failed with errors!" >&2
    exit 1
fi