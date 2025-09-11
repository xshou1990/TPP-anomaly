#!/bin/bash

# Initialize conda
eval "$(conda shell.bash hook)"

# Load configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"

# Verify environment
echo "Verifying conda environment..."
if ! conda activate tpp_env; then
    echo "Error: Failed to activate 'tpp_env' environment."
    exit 1
fi

# Check packages
echo "Environment activated. Checking packages..."
python -c "
try:
    import torch, pandas, numpy
    print('Required packages verified.')
except ImportError as e:
    print(f'Missing package: {e}')
    exit(1)
"

# Prepare directories
echo "Preparing directories..."
mkdir -p "$PREDICTIONS_DIR" || {
    echo "Error: Failed to create predictions directory"
    exit 1
}

# Run prediction
echo "Starting prediction with parameters:"
echo "---------------------------------"
echo "Model path:   $MODEL_DIR/$MODEL_FILE"
echo "Data path:    $DATA_PATH"
echo "Batch size:   $BATCH_SIZE"
echo "Num samples:  $NUM_SAMPLE"
echo "Look ahead:   $LOOK_AHEAD_TIME"
echo "Save to:      $PREDICTIONS_DIR/$PREDICTIONS_FILE"
echo "---------------------------------"

python predict.py \
    --model_path "$MODEL_DIR/$MODEL_FILE" \
    --data_path "$DATA_PATH" \
    --batch_size "$BATCH_SIZE" \
    --num_sample "$NUM_SAMPLE" \
    --look_ahead_time "$LOOK_AHEAD_TIME" \
    --save_predictions "$PREDICTIONS_DIR/$PREDICTIONS_FILE"

# Handle completion
if [ $? -eq 0 ]; then
    echo -e "\nPrediction completed successfully!"
    echo "Results saved to: $PREDICTIONS_DIR/$PREDICTIONS_FILE"
else
    echo -e "\nPrediction failed with errors!" >&2
    exit 1
fi