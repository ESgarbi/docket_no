# Docket Number Sequence Recognition Model

## Overview

This repository contains a custom-built deep learning model specifically designed to recognize and extract complete 13-digit docket number sequences from "Proof of Delivery" (Sales Order) documents used in Redebiz's automatic invoicing process. The model employs a specialized ResNet-based sequence recognition architecture and was trained from scratch (not a pre-trained model) to achieve high accuracy in identifying complete digit sequences in various document formats and qualities.

## Business Context

In Redebiz's operations, the accurate and efficient processing of "Proof of Delivery" documents is crucial for timely invoicing. Automating the extraction of complete docket number sequences from these documents significantly reduces manual data entry, accelerates the invoicing workflow, and minimizes human errors. This sequence recognition model serves as a key component in the document processing pipeline, enabling the system to:

1. Identify and locate docket number sequence fields within scanned documents
2. Accurately recognize the complete 13-digit sequences
3. Provide high-confidence sequence output for downstream systems
4. Process documents with varying quality, lighting, and formatting

## Model Architecture

The model uses a custom-designed `ResNetDigitSequence` architecture specifically optimized for sequence recognition with the following key components:

### Feature Extraction Backbone
- Modified ResNet architecture with residual blocks
- Three main convolutional layers with increasing channel depth
- Initial 7×7 convolution followed by batch normalization and max pooling
- Feature map dimensions progressively reduced while channel depth increases

### Sequence Recognition Head
- 13 specialized sequence position predictors, one for each digit in the sequence
- Enhanced network width for middle sequence positions (positions 8-12)
- Specialized feature learning with additional linear layers for challenging positions
- Adaptive dropout rates based on position complexity within the sequence

### Technical Specifications
- Input: RGB images (224×224)
- Output: Complete 13-digit sequence with 13 separate probability distributions
- Parameter count: ~3.2M
- Special attention to maintaining sequence consistency and contextual relationships

## Training Process

### Dataset
- Custom sequence dataset consisting of 13,126 training samples, 3,281 validation samples, and 1,000 test samples
- Each sample is a document image with a labeled 13-digit docket number sequence
- Images captured in various lighting conditions, angles, and document qualities

### Training Strategy
- **Loss Function**: Custom `DigitSequenceLoss` with:
  - Position-weighted cross-entropy losses across the entire sequence
  - Sequence consistency penalty using KL divergence between adjacent positions
  - Focal loss component for handling imbalanced classes within the sequence
- **Optimizer**: AdamW with weight decay 1e-4
- **Learning Rate**: Initial 5e-5 with OneCycleLR scheduler
- **Batch Size**: 16
- **Augmentation**: Subtle rotations, affine transformations, and color jitter
- **Regularization**: Varying dropout rates (0.2-0.3) based on position in sequence
- **Early Stopping**: After 15 epochs without improvement
- **Hardware**: Trained on Apple Silicon MPS

### Training Results
- **Per-Digit Accuracy**: 99.12%
- **Full Sequence Accuracy**: 91.40% (entire 13-digit sequence correctly recognized)
- **Training Time**: ~59 epochs (approximately 100 seconds per epoch)
- **Best Validation Loss**: 0.0454

## Usage in Production

The sequence recognition model is integrated into Redebiz's document processing pipeline, where it:
1. Receives pre-processed document images
2. Produces complete docket number sequence predictions with confidence scores
3. Outputs the full sequence along with individual position probabilities
4. Can flag low-confidence sequence predictions for human review

## Ongoing Evaluation Results

This section will be updated with sequence recognition performance metrics as the model is evaluated on new data in production.

| Date | Dataset | Per-Digit Accuracy | Full Sequence Accuracy | Report |
|------|---------|--------------------|-----------------------|--------|
| 2025-03-11 | Test (1000 samples) | 99.12% | 91.40% | [Report](evaluation_reports/evaluation_2025-03-11_13-15-09/evaluation_report.md) |
| 2025-03-11 | Test (1000 samples) | 99.47% | 95.20% | [Report](evaluation_reports/evaluation_2025-03-11_14-32-47/evaluation_report.md) |

## Prediction and Evaluation Tools

The repository includes several tools for making predictions and evaluating model performance:

### predict.py

A command-line tool for making predictions on single images using the trained model:

```bash
python predict.py --image path/to/image.png [--model_path path/to/model.pth] [--temp_scaler_path path/to/scaler.pth]
```

Options:
- `--image`: Path to the image file (required)
- `--model_path`: Path to the model file (default: outputs/final_model.pth)
- `--temp_scaler_path`: Path to the temperature scaler file (default: outputs/temperature_scaler.pth)
- `--output`: Path to save the JSON output (optional)
- `--no-calibration`: Disable confidence calibration with temperature scaling

The output is a JSON object containing:
- The predicted sequence
- Digit-by-digit predictions with confidence scores
- Overall sequence confidence metrics

### compare_models.py

A tool for side-by-side comparison of predictions from the original and calibrated models:

```bash
python compare_models.py --image path/to/image.png [--original_model path/to/original.pth] [--calibrated_model path/to/calibrated.pth]
```

This tool provides detailed comparison of:
- Predicted sequences from both models
- Digit-by-digit value and confidence comparisons
- Confidence differences between the models
- Visual comparison of confidence distributions

### batch_compare.py

A batch testing tool for comparing model performance across multiple images:

```bash
python batch_compare.py [--test_dir path/to/test/dir] [--num_samples 50]
```

This tool generates aggregate statistics including:
- Sequence match rate between models
- Digit match rate between models
- Position-specific error analysis
- Confidence calibration metrics

### eval.py

A comprehensive evaluation script for assessing model performance:

```bash
python eval.py [--model_path path/to/model.pth]
```

This script evaluates the model on the test dataset and generates a detailed report including:
- Per-digit and full sequence accuracy
- Confusion matrices for each digit position
- Wrong prediction analysis
- Confidence calibration metrics
- Sample predictions with visualizations

## Change History

| Date | Change | Description |
|------|--------|-------------|
| 2025-03-11 | Temperature Scaling Implementation | Implemented temperature scaling for confidence calibration, which effectively eliminated high-confidence errors. Temperature parameters were optimized for each digit position, resulting in more reliable confidence scores for model predictions. This change enables better decision-making in the production pipeline by ensuring confidence scores accurately reflect the model's true certainty. |
| 2025-03-11 | Label Smoothing and Focal Loss Training | Trained a new model using label smoothing (0.05) and focal loss (gamma=1.0) to reduce overconfidence and focus on hard examples during training. This reduced total errors by 39.5% and improved sequence accuracy by 3.8%, making the model more robust overall. |
| 2025-03-11 | Prediction Tool Enhancements | Updated prediction tools to support different models and confidence calibration. Added compare_models.py for side-by-side comparisons and batch_compare.py for batch testing. These tools enable easy comparison between original and calibrated models. |

## Model Performance Comparison

### Original vs. Calibrated Model

Batch testing on 50 random test images showed:
- **Sequence Match Rate**: 96.00% between the models (48/50 images match)
- **Digit Match Rate**: 99.69% between the models (648/650 digits match)
- **Average Confidence Difference**: +0.0013 (calibrated model has slightly higher confidence overall)

The calibrated model provides several key advantages:
1. **Better Confidence Calibration**: More reliable confidence scores that better reflect the model's certainty
2. **Eliminated High-Confidence Errors**: Temperature scaling completely eliminated high-confidence errors
3. **Improved Accuracy**: Overall sequence accuracy improved from 91.40% to 95.20%
4. **More Robust Predictions**: Lower variation in confidence across different samples

### Recommendations for Use

- **For Regular Use**: Use the calibrated model (`outputs/calibrated_model.pth`) as the default
- **For Critical Applications**: Use confidence thresholds based on calibrated scores (e.g., flag predictions with <0.95 confidence)
- **For Production Integration**: Always apply temperature scaling during inference for reliable confidence scores

## Future Improvements

- Fine-tuning on more diverse document types and sequence variations
- Exploring attention mechanisms for better contextual understanding of sequences
- Implementing active learning to improve on challenging sequence patterns
- Investigating sequence modeling alternatives to enhance full-sequence accuracy

## Acknowledgments

This sequence recognition model was developed by the Redebiz AI team to address specific business needs in invoice processing. The architecture and training approach were custom-designed for this particular sequence recognition application and not derived from pre-existing models.