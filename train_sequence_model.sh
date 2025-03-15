#!/bin/bash
# Script to train the digit sequence recognition model

# Default values
PRETRAINED_MODEL=""
DATA_DIR="/Users/erick/git/prod/docket_no/sequence/001/"
MODEL_PATH="/Users/erick/git/prod/docket_no/outputs/final_model.pth"
EPOCHS=160
LEARNING_RATE=0.00005
LABEL_SMOOTHING=0.1
FOCAL_GAMMA=2.0
CHECKPOINT_EVERY=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      PRETRAINED_MODEL="$2"
      shift 2
      ;;
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --label_smoothing)
      LABEL_SMOOTHING="$2"
      shift 2
      ;;
    --focal_gamma)
      FOCAL_GAMMA="$2"
      shift 2
      ;;
    --checkpoint_every)
      CHECKPOINT_EVERY="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create the output directories if they don't exist
mkdir -p /Users/erick/git/prod/docket_no/outputs
mkdir -p /Users/erick/git/prod/docket_no/outputs/checkpoints
mkdir -p /Users/erick/git/prod/docket_no/logs

# Set the Python executable - use python3 if available
if command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

# If using pretrained weights, show that in the output
if [ -n "$PRETRAINED_MODEL" ]; then
    echo "Using pretrained weights from: $PRETRAINED_MODEL"
    MODEL_ARG="$PRETRAINED_MODEL"
else
    echo "Starting with fresh model (no pretrained weights)"
    # Remove previous models to start fresh
    rm -f /Users/erick/git/prod/docket_no/outputs/final_model.pth
    MODEL_ARG="$MODEL_PATH"
fi

# Show training configuration
echo "Training with the following configuration:"
echo "Data directory: $DATA_DIR"
echo "Model path: $MODEL_ARG"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Label smoothing: $LABEL_SMOOTHING"
echo "Focal gamma: $FOCAL_GAMMA"
echo "Checkpoint every: $CHECKPOINT_EVERY epochs"

# Updated training command with supported parameters
$PYTHON /Users/erick/git/prod/docket_no/google_colab_training_resnet.py \
    --calibrate \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL_ARG" \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --label_smoothing $LABEL_SMOOTHING \
    --focal_gamma $FOCAL_GAMMA \
    --checkpoint_every $CHECKPOINT_EVERY

# Exit with the status of the Python script
exit $? 