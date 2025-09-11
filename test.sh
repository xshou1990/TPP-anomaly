#!/bin/bash

# Initialize conda
eval "$(conda shell.bash hook)"

# Load central configuration
source ./config.sh

# Configuration
export MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
export SAVE_RESULTS="${RESULTS_DIR}/${RESULTS_FILE}"

# Activate environment
echo "Verifying conda environment..."
if ! conda activate tpp_env; then
    echo "Error: Failed to activate 'tpp_env' environment."
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

# Create directories
echo "Preparing directories..."
mkdir -p "${RESULTS_DIR}" || {
    echo "Error: Failed to create results directory"
    exit 1
}

# Run testing
echo "Starting testing..."
echo "-----------------------------"
echo "Model path:  ${MODEL_PATH}"
echo "Data path:   ${DATA_PATH}"
echo "Batch size:  ${BATCH_SIZE}"
echo "Results:     ${SAVE_RESULTS}"
echo "-----------------------------"

python test.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --save_results "${SAVE_RESULTS}"

if [ $? -eq 0 ]; then
    echo -e "\nTesting completed successfully!"
    echo "Results saved to: ${SAVE_RESULTS}"
else
    echo -e "\nTesting failed with errors!" >&2
    exit 1
fi