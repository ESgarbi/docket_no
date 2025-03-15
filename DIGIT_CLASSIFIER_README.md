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

| Date | Dataset | Per-Digit Accuracy | Full Sequence Accuracy | Notes |
|------|---------|--------------------|-----------------------|-------|
|      |         |                    |                       |       |

## Future Improvements

- Fine-tuning on more diverse document types and sequence variations
- Exploring attention mechanisms for better contextual understanding of sequences
- Implementing active learning to improve on challenging sequence patterns
- Investigating sequence modeling alternatives to enhance full-sequence accuracy

## Acknowledgments

This sequence recognition model was developed by the Redebiz AI team to address specific business needs in invoice processing. The architecture and training approach were custom-designed for this particular sequence recognition application and not derived from pre-existing models.
