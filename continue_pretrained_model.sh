#!/bin/bash

# Script to continue pretraining from the last checkpoint to maximize sequence accuracy

# Configuration variables
SYNTHETIC_DATA_DIR="/Users/erick/git/prod/docket_no/data/synthetic"
TRAIN_DIR="/Users/erick/git/prod/docket_no/data/synthetic/train"
EVAL_DIR="/Users/erick/git/prod/docket_no/data/synthetic/eval"
LAST_CHECKPOINT="/Users/erick/git/prod/docket_no/outputs/checkpoints/pretrained_model_checkpoint_epoch_30.pth"
OUTPUT_MODEL_PATH="/Users/erick/git/prod/docket_no/outputs/pretrained_model_final.pth"
CHECKPOINT_DIR="/Users/erick/git/prod/docket_no/outputs/checkpoints"
LOG_DIR="/Users/erick/git/prod/docket_no/logs/pretrain_continued"

# Training hyperparameters - Adjusted for final convergence
LEARNING_RATE=0.0001   # Reduced learning rate for fine-tuning
NUM_EPOCHS=40          # More epochs
LABEL_SMOOTHING=0.03   # Reduced label smoothing for better certainty
FOCAL_GAMMA=1.0        # Reduced focal gamma to focus more on harder examples
CHECKPOINT_EVERY=5     # Save checkpoints every 5 epochs

echo "========================================================"
echo "  Continuing Pretraining from Last Checkpoint"
echo "========================================================"
echo "Starting from checkpoint: $LAST_CHECKPOINT"
echo "Synthetic data directory: $SYNTHETIC_DATA_DIR"
echo "Output pretrained model: $OUTPUT_MODEL_PATH"
echo "Label smoothing: $LABEL_SMOOTHING"
echo "Focal loss gamma: $FOCAL_GAMMA"
echo "Learning rate: $LEARNING_RATE"
echo "Training epochs: $NUM_EPOCHS"
echo "========================================================"

# Create the output directories if they don't exist
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

# Run the continued training with synthetic data
echo "Starting continued training process..."
$PYTHON /Users/erick/git/prod/docket_no/google_colab_training_resnet.py \
  --calibrate \
  --data_dir "$SYNTHETIC_DATA_DIR" \
  --model_path "$LAST_CHECKPOINT" \
  --label_smoothing $LABEL_SMOOTHING \
  --focal_gamma $FOCAL_GAMMA \
  --learning_rate $LEARNING_RATE \
  --epochs $NUM_EPOCHS \
  --checkpoint_every $CHECKPOINT_EVERY

# Copy the final model to the output path
cp "$LAST_CHECKPOINT" "$OUTPUT_MODEL_PATH"

echo "Continued training completed! The final pretrained model has been saved to $OUTPUT_MODEL_PATH."
echo "Final model sequence accuracy should now be higher."
echo ""
echo "You can now use this pretrained model as a starting point for training with sequence data:"
echo "bash train_sequence_model.sh --model_path $OUTPUT_MODEL_PATH"

# Exit with the status of the Python script
exit $?
