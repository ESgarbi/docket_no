#!/bin/bash

# This script trains a model with confidence calibration techniques to reduce high-confidence errors
# It uses label smoothing and focal loss to prevent overconfidence during training

# Configuration - edit these values as needed
DATA_DIR="/Users/erick/git/prod/docket_no/sequence/001"
MODEL_PATH="/Users/erick/git/prod/docket_no/outputs/final_model.pth" # Set to empty string for new model
OUTPUT_MODEL="/Users/erick/git/prod/docket_no/outputs/calibrated_model.pth"
LABEL_SMOOTHING=0.05      # Reduced from 0.1 to 0.05 for stability
FOCAL_GAMMA=1.0           # Reduced from 2.0 to 1.0 for stability
LEARNING_RATE=0.000001    # Much lower learning rate (1e-6) for stability
NUM_EPOCHS=10s0              # Reduced epochs for quicker training

echo "========================================================"
echo "  Training with Confidence Calibration Techniques"
echo "========================================================"
echo "Data directory: $DATA_DIR"
echo "Starting model: $MODEL_PATH"
echo "Output model: $OUTPUT_MODEL"
echo "Label smoothing: $LABEL_SMOOTHING"
echo "Focal loss gamma: $FOCAL_GAMMA"
echo "Learning rate: $LEARNING_RATE"
echo "Training epochs: $NUM_EPOCHS"
echo "========================================================"

# Run the training with confidence calibration
python google_colab_training_resnet.py \
  --calibrate \
  --data_dir $DATA_DIR \
  --model_path $MODEL_PATH \
  --label_smoothing $LABEL_SMOOTHING \
  --focal_gamma $FOCAL_GAMMA \
  --learning_rate $LEARNING_RATE \
  --epochs $NUM_EPOCHS

echo "Training completed! The calibrated model has been saved."
echo "You can now evaluate it using: python eval.py" 