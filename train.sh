#!/bin/bash

# Initialize conda
eval "$(conda shell.bash hook)"

# This script runs the RMTPP training with proper environment handling

# --------------------------
# Configuration Section
# --------------------------
EMBED_DIM=32 #64 for best time prediction
HIDDEN_DIM=64 #128 for best time prediction
BATCH_SIZE=256 #32 for best time prediction
EPOCHS=30
NUM_LAYERS=2 
LEARNING_RATE=0.001
DATA_PATH="./Datasets/amazon/"
SAVE_DIR="./models"

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
mkdir -p "${SAVE_DIR}" || {
    echo "Error: Failed to create save directory ${SAVE_DIR}"
    exit 1
}

# --------------------------
# Run Training
# --------------------------
echo "Starting training with parameters:"
echo "---------------------------------"
echo "Embed dim:   ${EMBED_DIM}"
echo "Hidden dim:  ${HIDDEN_DIM}"
echo "Batch size:  ${BATCH_SIZE}"
echo "Num layers:  ${NUM_LAYERS}"
echo "Epochs:      ${EPOCHS}"
echo "LR:          ${LEARNING_RATE}"
echo "Data path:   ${DATA_PATH}"
echo "Save dir:    ${SAVE_DIR}"
echo "---------------------------------"

python train.py \
    --embed_dim "${EMBED_DIM}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --batch_size "${BATCH_SIZE}" \
    --num_layers "${NUM_LAYERS}" \
    --epochs "${EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --data_path "${DATA_PATH}" \
    --save_dir "${SAVE_DIR}"

# --------------------------
# Completion Handling
# --------------------------
if [ $? -eq 0 ]; then
    echo -e "\nTraining completed successfully!"
    echo "Model checkpoints saved to: ${SAVE_DIR}"
else
    echo -e "\nTraining failed with errors!" >&2
    exit 1
fi
