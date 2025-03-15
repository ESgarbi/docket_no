#!/bin/bash

# Script to pretrain a model using synthetic data and generate weights for future training

# Configuration variables
SYNTHETIC_DATA_DIR="/Users/erick/git/prod/docket_no/data/synthetic"
TRAIN_DIR="/Users/erick/git/prod/docket_no/data/synthetic/train"
EVAL_DIR="/Users/erick/git/prod/docket_no/data/synthetic/eval"
OUTPUT_MODEL_PATH="/Users/erick/git/prod/docket_no/outputs/pretrained_model.pth"
CHECKPOINT_DIR="/Users/erick/git/prod/docket_no/outputs/checkpoints"
LOG_DIR="/Users/erick/git/prod/docket_no/logs/pretrain"

# Training hyperparameters
LEARNING_RATE=0.0005
NUM_EPOCHS=30
LABEL_SMOOTHING=0.05
FOCAL_GAMMA=1.5
CHECKPOINT_EVERY=5

echo "========================================================"
echo "  Pretraining Model with Synthetic Data"
echo "========================================================"
echo "Synthetic data directory: $SYNTHETIC_DATA_DIR"
echo "Output pretrained model: $OUTPUT_MODEL_PATH"
echo "Label smoothing: $LABEL_SMOOTHING"
echo "Focal loss gamma: $FOCAL_GAMMA"
echo "Learning rate: $LEARNING_RATE"
echo "Training epochs: $NUM_EPOCHS"
echo "========================================================"

# Create the output and data directories if they don't exist
mkdir -p "$(dirname "$OUTPUT_MODEL_PATH")"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$TRAIN_DIR"
mkdir -p "$EVAL_DIR"

# Set the Python executable - use python3 if available
if command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

# Create a simplified dataset structure with train/eval split
echo "Setting up dataset structure for training..."

# Create a temporary Python script to split the dataset
cat > /tmp/split_synthetic_data.py << 'EOF'
import os
import json
import shutil
import random
from pathlib import Path

# Paths
SYNTHETIC_DIR = "/Users/erick/git/prod/docket_no/data/synthetic"
TRAIN_DIR = "/Users/erick/git/prod/docket_no/data/synthetic/train"
EVAL_DIR = "/Users/erick/git/prod/docket_no/data/synthetic/eval"

# Load the labels file
with open(os.path.join(SYNTHETIC_DIR, "labels.json"), "r") as f:
    data = json.load(f)

# Split data into train (80%) and eval (20%)
random.seed(42)  # For reproducibility
random.shuffle(data)
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
eval_data = data[split_idx:]

print(f"Total records: {len(data)}")
print(f"Training set: {len(train_data)} records")
print(f"Evaluation set: {len(eval_data)} records")

# Create train directory and copy files
os.makedirs(TRAIN_DIR, exist_ok=True)
with open(os.path.join(TRAIN_DIR, "labels.json"), "w") as f:
    json.dump(train_data, f)

# Create eval directory and copy files
os.makedirs(EVAL_DIR, exist_ok=True)
with open(os.path.join(EVAL_DIR, "labels.json"), "w") as f:
    json.dump(eval_data, f)

# Create symbolic links to all image files in both directories
# This avoids duplicating large image files
for image_file in Path(SYNTHETIC_DIR).glob("*.png"):
    # Create symbolic links in both train and eval dirs
    os.symlink(image_file, os.path.join(TRAIN_DIR, os.path.basename(image_file)))
    os.symlink(image_file, os.path.join(EVAL_DIR, os.path.basename(image_file)))

print("Dataset preparation complete!")
EOF

# Execute the temporary Python script
echo "Splitting synthetic data into train/eval sets..."
$PYTHON /tmp/split_synthetic_data.py

# Run the pretraining with synthetic data
echo "Starting pretraining process..."
$PYTHON /Users/erick/git/prod/docket_no/google_colab_training_resnet.py \
  --calibrate \
  --data_dir "$SYNTHETIC_DATA_DIR" \
  --model_path "$OUTPUT_MODEL_PATH" \
  --label_smoothing $LABEL_SMOOTHING \
  --focal_gamma $FOCAL_GAMMA \
  --learning_rate $LEARNING_RATE \
  --epochs $NUM_EPOCHS \
  --checkpoint_every $CHECKPOINT_EVERY

echo "Pretraining completed! The pretrained model has been saved to $OUTPUT_MODEL_PATH."
echo "You can now use this pretrained model as a starting point for training with sequence data."
echo "Example command:"
echo "bash train_sequence_model.sh --model_path $OUTPUT_MODEL_PATH"

# Cleanup
rm -f /tmp/split_synthetic_data.py

# Exit with the status of the Python script
exit $?
