#!/bin/bash

# Central configuration file for RMTPP Taxi project

# --------------------------
# Training Configuration
# --------------------------
export EMBED_DIM=32
export HIDDEN_DIM=16
export BATCH_SIZE=256
export EPOCHS=200
export NUM_LAYERS=2
export LEARNING_RATE=0.001
export TIME_EMBED_SIZE=16
export NUM_HEADS=2
export MC_SAMPLES=20
export INTEGRAL_SAMPLES=20

# --------------------------
# Path Configuration
# --------------------------
export DATA_PATH="./Datasets/taxi/"
export MODEL_DIR="./models"
export RESULTS_DIR="./results"
export PREDICTIONS_DIR="./predictions"

# Model filenames
export MODEL_FILE="rmtpp_taxi_best.pt"
export RESULTS_FILE="taxi_predictions.pkl"
export PREDICTIONS_FILE="JJJRMTPP_Pred.pkl"

# --------------------------
# Thinning Configuration (for prediction)
# --------------------------
export NUM_SEQUENCE=10
export NUM_SAMPLE=1
export NUM_EXP=500
export LOOK_AHEAD_TIME=10
export PATIENCE_COUNTER=5
export OVER_SAMPLE_RATE=5
export NUM_SAMPLES_BOUNDARY=5
export DTIME_MAX=5
export NUM_STEP_GEN=10