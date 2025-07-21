#!/bin/bash

# Initialize conda
eval "$(conda shell.bash hook)"

# Configuration
MODEL_PATH="./models/model_epoch_30.pt"
DATA_PATH="./Datasets/amazon/test.pkl"
OUTPUT_PATH="./results/predictions_results_custom.csv"
NUM_EVENT_TYPES=17
EMBED_DIM=64
HIDDEN_DIM=128
BATCH_SIZE=32
MIN_SEQ_LENGTH=2  # Minimum sequence length to process

# Verify environment
echo "Activating conda environment..."
conda activate tpp_env || {
    echo "Failed to activate tpp_env"
    exit 1
}

# Create output directory
mkdir -p "./results"

# Run prediction with strict filtering
echo "Starting prediction with sequence validation..."
python predict.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --num_event_types "$NUM_EVENT_TYPES" \
    --embed_dim "$EMBED_DIM" \
    --hidden_dim "$HIDDEN_DIM" \
    --batch_size "$BATCH_SIZE" \
    --min_sequence_length "$MIN_SEQ_LENGTH"

if [ $? -eq 0 ]; then
    echo -e "\nPrediction completed successfully!"
    echo "Results saved to: $OUTPUT_PATH"
    # Show sample output
    echo -e "\nFirst 5 predictions:"
    head -n 5 "$OUTPUT_PATH"
else
    echo -e "\nPrediction failed!" >&2
    exit 1
fi