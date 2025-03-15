"""
Comprehensive training script for Google Colab with TensorBoard integration.
This file contains all the necessary code to train the digit sequence recognition model.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import timm
import time
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Optional
import io
import torch.optim as optim
import sys
import random
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
import math

# Constants
NUM_DIGITS = 13
NUM_CLASSES = 10

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetDigitSequence(nn.Module):
    def __init__(self, lstm_hidden_size=256, num_lstm_layers=2):
        super(ResNetDigitSequence, self).__init__()
        self.in_channels = 64
        
        # ResNet Feature Extractor
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        
        # Feature extractor for each digit region
        self.digit_extractors = nn.ModuleList()
        for _ in range(13):
            self.digit_extractors.append(nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            ))
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Digit classifiers
        self.digit_classifiers = nn.ModuleList()
        for _ in range(13):
            self.digit_classifiers.append(nn.Sequential(
                nn.Linear(lstm_hidden_size * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 10)
            ))
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Shared feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # [batch_size, 256, H, W]
        
        # Manual splitting into 13 regions
        batch_size = x.size(0)
        width = x.size(3)
        region_width = width // 13
        
        # Extract features for each region
        digit_features = []
        for i in range(13):
            start = i * region_width
            end = min((i + 1) * region_width, width)
            
            if start >= width:  # Safety check
                # Use the last valid region if dimensions don't divide evenly
                region = x[:, :, :, (width - region_width):width]
            else:
                region = x[:, :, :, start:end]
            
            # Process with the region-specific extractor
            region_features = self.digit_extractors[i](region)
            region_features = torch.flatten(region_features, 1)
            digit_features.append(region_features)
        
        # Stack to create sequence [batch_size, 13, 128]
        sequence = torch.stack(digit_features, dim=1)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(sequence)
        
        # Apply digit classifiers
        digit_outputs = {}
        for i, classifier in enumerate(self.digit_classifiers):
            digit_outputs[f'digit_{i+1}'] = classifier(lstm_out[:, i, :])
        
        return digit_outputs
    # def __init__(self):
    #     super(ResNetDigitSequence, self).__init__()
    #     self.in_channels = 64
        
    #     self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #     self.bn1 = nn.BatchNorm2d(64)
    #     self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    #     # Create ResNet layers
    #     self.layer1 = self._make_layer(64, 2, stride=1)
    #     self.layer2 = self._make_layer(128, 2, stride=2)
    #     self.layer3 = self._make_layer(256, 2, stride=2)
        
    #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    #     self.flatten_dim = 256
        
    #     # Create 13 separate digit classifiers with extra capacity for middle digits
    #     self.digit_classifiers = nn.ModuleList()
    #     for i in range(13):
    #         if 7 <= i <= 11 and i != 9 and i != 10 and i != 11 and i != 12 and i != 13:  # Middle digits (8-12) and (2-4) <ej: 9, 10, 11, 12, 13>
    #             self.digit_classifiers.append(nn.Sequential(
    #                 nn.Linear(self.flatten_dim, 256),  # Increase from 128 to 256
    #                 nn.ReLU(),
    #                 nn.Dropout(0.3),  # Slightly higher dropout
    #                 nn.Linear(256, 128),  # Add an extra layer
    #                 nn.ReLU(),
    #                 nn.Dropout(0.2),
    #                 nn.Linear(128, 10)
    #             ))
    #         else:
    #             self.digit_classifiers.append(nn.Sequential(
    #                 nn.Linear(self.flatten_dim, 128),
    #                 nn.ReLU(),
    #                 nn.Dropout(0.2),
    #                 nn.Linear(128, 10)
    #             ))
        
    # def _make_layer(self, out_channels, num_blocks, stride):
    #     strides = [stride] + [1] * (num_blocks - 1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(ResidualBlock(self.in_channels, out_channels, stride))
    #         self.in_channels = out_channels
    #     return nn.Sequential(*layers)
        
    # def forward(self, x):
    #     # Extract features
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = self.pool(x)
        
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
        
    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
        
    #     # Apply each digit classifier
    #     digit_outputs = {}
    #     for i, classifier in enumerate(self.digit_classifiers):
    #         digit_outputs[f'digit_{i+1}'] = classifier(x)
            
    #     return digit_outputs

class DigitSequenceLoss(nn.Module):
    """
    Custom loss function for 13-digit sequence recognition.
    Calculates separate losses for each digit position and allows for weighting.
    
    Features:
    - Per-digit position weighting
    - Sequence consistency penalty using KL divergence
    - Focal loss to focus on hard examples
    - Label smoothing to prevent overconfidence
    """
    def __init__(self, num_digits=13, weights=None, use_sequence_penalty=False, 
                 sequence_lambda=0.1, focal_gamma=2.0, label_smoothing=0.1):
        """
        Initialize the DigitSequenceLoss.
        
        Args:
            num_digits (int): Number of digits in the sequence (default: 13)
            weights (list): Weights for each digit position loss (default: equal weights)
            use_sequence_penalty (bool): Whether to add a sequence consistency penalty
            sequence_lambda (float): Weight for the sequence consistency penalty
            focal_gamma (float): Gamma parameter for focal loss (0 = standard CE loss)
                               Higher values (2-5) focus more on hard examples
            label_smoothing (float): Amount of label smoothing (0-1)
                                  Higher values reduce model confidence
        """
        super(DigitSequenceLoss, self).__init__()
        self.num_digits = num_digits
        self.use_sequence_penalty = use_sequence_penalty
        self.sequence_lambda = sequence_lambda
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        # Default: equal weights for all digit positions
        if weights is None:
            self.weights = [1.0] * num_digits
        else:
            assert len(weights) == num_digits, "Weights must match number of digits"
            self.weights = weights
            
        # Normalize weights to sum to number of digits
        weight_sum = sum(self.weights)
        self.weights = [w * self.num_digits / weight_sum for w in self.weights]
        
    def apply_label_smoothing(self, targets, num_classes=10):
        """
        Convert one-hot targets to smoothed targets.
        Label smoothing helps prevent the model from becoming too confident
        by distributing some probability mass to non-target classes.
        
        Args:
            targets: Target digit indices of shape (batch_size,)
            num_classes: Number of possible digit values (0-9)
            
        Returns:
            Smoothed target probabilities of shape (batch_size, num_classes)
        """
        batch_size = targets.size(0)
        
        # Create a numerically stable version with small epsilon
        epsilon = 1e-6
        smoothing_value = max(self.label_smoothing, epsilon)
        
        # Calculate smoothed values with bounds
        smoothed_targets = torch.zeros(batch_size, num_classes, device=targets.device)
        
        # Fill with smoothing value
        smoothed_targets.fill_(smoothing_value / (num_classes - 1))
        
        # Fill in the correct target with remaining probability mass
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing_value)
        
        return smoothed_targets
        
    def forward(self, outputs, targets):
        """
        Calculate weighted loss across all digit positions.
        
        Args:
            outputs (dict): Dictionary of logits for each digit position, 
                          keys should be 'digit_1', 'digit_2', etc.
            targets (torch.Tensor): Target digit indices of shape (batch_size, num_digits)
            
        Returns:
            tuple: (total_loss, digit_losses_dict)
                - total_loss: weighted average of all digit losses
                - digit_losses_dict: dictionary of individual digit losses
        """
        total_loss = 0.0
        digit_losses = {}
        all_probs = []
        
        digit_positions = [f'digit_{i+1}' for i in range(self.num_digits)]
        
        # Apply small epsilon for numerical stability
        eps = 1e-8
        
        for i, pos in enumerate(digit_positions):
            if pos in outputs:
                # Get predictions and targets for this digit position
                digit_preds = outputs[pos]
                digit_targets = targets[:, i]
                
                try:
                    # Enhanced loss calculation with label smoothing and focal loss
                    if self.label_smoothing > 0:
                        # Apply label smoothing
                        smooth_targets = self.apply_label_smoothing(digit_targets, num_classes=digit_preds.size(1))
                        
                        # Calculate KL divergence between smoothed targets and predictions
                        log_probs = F.log_softmax(digit_preds, dim=1)
                        # Clip log probs for stability
                        log_probs = torch.clamp(log_probs, min=-100.0, max=100.0)
                        
                        # Calculate loss with clipping to prevent extreme values
                        loss = -(smooth_targets * log_probs).sum(dim=1)
                        
                        # Apply focal weighting if enabled
                        if self.focal_gamma > 0:
                            # Get probability of the target class
                            probs = torch.softmax(digit_preds, dim=1)
                            target_probs = torch.gather(
                                probs, 
                                dim=1, 
                                index=digit_targets.unsqueeze(1)
                            ).squeeze(1)
                            
                            # Add epsilon for numerical stability
                            target_probs = torch.clamp(target_probs, min=eps, max=1.0)
                            
                            # Apply focal weighting: focus on hard examples
                            focal_weight = (1 - target_probs) ** self.focal_gamma
                            focal_weight = torch.clamp(focal_weight, min=0.0, max=100.0)
                            loss = loss * focal_weight
                        
                        digit_loss = loss.mean()
                    else:
                        # Standard cross entropy with optional focal loss
                        ce_loss = F.cross_entropy(digit_preds, digit_targets, reduction='none')
                        
                        # Apply focal loss if gamma > 0
                        if self.focal_gamma > 0:
                            probs = torch.gather(
                                F.softmax(digit_preds, dim=1), 
                                dim=1, 
                                index=digit_targets.unsqueeze(1)
                            ).squeeze(1)
                            
                            # Add epsilon for numerical stability
                            probs = torch.clamp(probs, min=eps, max=1.0)
                            
                            focal_weight = (1 - probs) ** self.focal_gamma
                            focal_weight = torch.clamp(focal_weight, min=0.0, max=100.0)
                            digit_loss = (focal_weight * ce_loss).mean()
                        else:
                            digit_loss = ce_loss.mean()
                    
                    # Apply weight for this position
                    weighted_loss = digit_loss * self.weights[i]
                    
                    # Store individual loss for reporting
                    digit_losses[pos] = digit_loss.item()
                    
                    # Add to total loss
                    total_loss += weighted_loss
                    
                    # Store probabilities for sequence penalty
                    if self.use_sequence_penalty:
                        all_probs.append(F.softmax(digit_preds, dim=1))
                except Exception as e:
                    print(f"Error in loss calculation for {pos}: {e}")
                    # Fall back to standard cross entropy
                    digit_loss = F.cross_entropy(digit_preds, digit_targets)
                    weighted_loss = digit_loss * self.weights[i]
                    digit_losses[pos] = digit_loss.item()
                    total_loss += weighted_loss
                    
                    if self.use_sequence_penalty:
                        all_probs.append(F.softmax(digit_preds, dim=1))
        
        # Apply sequence consistency penalty if enabled
        if self.use_sequence_penalty and len(all_probs) > 1:
            sequence_penalty = 0.0
            
            # Calculate pairwise probability distribution differences between adjacent positions
            for i in range(len(all_probs) - 1):
                # KL divergence to penalize dramatically different distributions
                # between adjacent digit positions
                try:
                    # Add small epsilon to avoid log(0)
                    p_stable = all_probs[i] + eps
                    q_stable = all_probs[i+1] + eps
                    
                    # Normalize to ensure they sum to 1
                    p_stable = p_stable / p_stable.sum(dim=1, keepdim=True)
                    q_stable = q_stable / q_stable.sum(dim=1, keepdim=True)
                    
                    sequence_penalty += F.kl_div(
                        torch.log(p_stable), q_stable, 
                        reduction='batchmean'
                    )
                except Exception as e:
                    print(f"Error in sequence penalty calculation: {e}")
                    # Skip this pair if there's an error
                    continue
            
            # Add sequence penalty to total loss
            if len(all_probs) > 1:
                sequence_penalty = sequence_penalty / (len(all_probs) - 1)
                sequence_penalty = torch.clamp(sequence_penalty, min=0.0, max=10.0)
                total_loss = total_loss + self.sequence_lambda * sequence_penalty
                
                # Store sequence penalty for reporting
                digit_losses['sequence_penalty'] = sequence_penalty.item()
        
        # Return both the total loss and individual losses
        return total_loss / self.num_digits, digit_losses

# ============================================================================
# DATASET DEFINITION
# ============================================================================

class DigitSequenceDataset(Dataset):
    """
    Dataset for training the digit sequence recognition model.
    
    Args:
        image_paths: List of paths to the images
        labels: List of labels, where each label is a string of 13 digits
        transform: Transformations to apply to the images
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')  # Load as grayscale
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            # Convert label string to tensor
            label = torch.tensor([int(digit) for digit in self.labels[idx]], dtype=torch.long)
            return image, label
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a dummy sample in case of error
            if self.transform:
                dummy_image = torch.zeros((1, 224, 224))
            else:
                dummy_image = Image.new('L', (224, 224), 0)
                if self.transform:
                    dummy_image = self.transform(dummy_image)
            
            dummy_label = torch.zeros(13, dtype=torch.long)
            return dummy_image, dummy_label

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_transforms():
    """
    Get transforms for training and validation datasets
    
    Returns:
        dict: Dictionary with 'train' and 'val' transforms
    """
    # Enhanced transformations for training with better augmentation
    train_transform = transforms.Compose([
        # More subtle but varied augmentations
        transforms.RandomRotation(3),  # More subtle rotation
        transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Slight affine transformations
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])
    # Transformations for validation (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalization for grayscale
    ])
    
    return {'train': train_transform, 'val': val_transform}

def visualize_sample(image_path, label, writer=None, step=0, save_dir=None):
    """
    Visualize a sample image with its label.
    
    Args:
        image_path: Path to the image
        label: Ground truth label
        writer: TensorBoard SummaryWriter (optional)
        step: Step/iteration for TensorBoard logging
        save_dir: Directory to save the visualization
    """
    image = plt.imread(image_path)
    plt.figure(figsize=(10, 5))
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    
    if writer is not None:
        # Convert matplotlib figure to image for TensorBoard
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = transforms.ToTensor()(img)
        writer.add_image('Sample Image', img_tensor, step)
    
    # Save the figure if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'sample_image_step_{step}.png'))
    
    plt.close()

def plot_confusion_matrix(cm, class_names, writer=None, step=0, title='Confusion Matrix', save_dir=None):
    """
    Plot confusion matrix and optionally log to TensorBoard.
    
    Args:
        cm: Confusion matrix
        class_names: Class names
        writer: TensorBoard SummaryWriter (optional)
        step: Step/iteration for TensorBoard logging
        title: Title for the plot
        save_dir: Directory to save the visualization
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations for each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if writer is not None:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = transforms.ToTensor()(img)
        writer.add_image(f'{title}', img_tensor, step)
    
    # Save the figure if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_title = title.replace(' ', '_').replace('-', '_')
        plt.savefig(os.path.join(save_dir, f'{safe_title}_epoch_{step}.png'))
    
    plt.close()

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, 
                scheduler_type='cosine', patience=5, min_lr=1e-6, log_dir='runs',
                criterion=None, optimizer=None, device=None):
    """
    Train the digit sequence recognition model.
    
    Args:
        model: The DigitSequenceModel to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        scheduler_type: Type of learning rate scheduler to use ('plateau', 'cosine', 'onecycle', or None)
        patience: Number of epochs to wait for improvement before early stopping
        min_lr: Minimum learning rate for schedulers
        log_dir: Directory for TensorBoard logs
        criterion: Loss function to use (default: DigitSequenceLoss)
        optimizer: Optimizer to use (default: AdamW)
        device: Device to use for training (default: auto-detected)
        
    Returns:
        tuple: (history dict, trained model)
    """
    # Set up device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                             'mps' if torch.backends.mps.is_available() else 
                             'cpu')
    
    print(f"Training on device: {device}")
    model = model.to(device)
    
    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create loss function if not provided
    if criterion is None:
        criterion = DigitSequenceLoss(num_digits=13)
    
    # Create optimizer if not provided
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Create scheduler
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=min_lr
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=min_lr
        )
    elif scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, total_steps=num_epochs * len(train_loader)
        )
    else:
        scheduler = None
        
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'digit_acc': [],
        'sequence_acc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss, digit_losses = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            train_loss += loss.item()
            avg_loss = train_loss / (batch_idx + 1)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
            
            # Update OneCycle scheduler if used
            if scheduler_type == 'onecycle' and scheduler is not None:
                scheduler.step()
        
        # Calculate average training loss for this epoch
        train_loss /= len(train_loader)
        
        # Validation phase
        if val_loader is not None:
            val_loss, digit_acc, sequence_acc = validate(model, val_loader, criterion, device)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['digit_acc'].append(digit_acc)
            history['sequence_acc'].append(sequence_acc)
            
            # Log to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/digit', digit_acc, epoch)
            writer.add_scalar('Accuracy/sequence', sequence_acc, epoch)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Digit Acc: {digit_acc:.2f}%, "
                  f"Sequence Acc: {sequence_acc:.2f}%")
            
            # Update schedulers that step after validation
            if scheduler_type == 'plateau' and scheduler is not None:
                scheduler.step(val_loss)
            elif scheduler_type == 'cosine' and scheduler is not None:
                scheduler.step()
                
            # Check if this is the best model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict().copy()
                epochs_no_improve = 0
                
                # Save best model
                torch.save(best_model, os.path.join(log_dir, 'best_model.pth'))
            else:
                epochs_no_improve += 1
                
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            # If no validation set, just update history with training metrics
            history['train_loss'].append(train_loss)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
            
            # Update scheduler if applicable
            if scheduler_type == 'cosine' and scheduler is not None:
                scheduler.step()
                
    # Load best model if validation was used
    if val_loader is not None and best_model is not None:
        model.load_state_dict(best_model)
        
    # Close TensorBoard writer
    writer.close()
    
    return history, model

def validate(model, val_loader, criterion, device=None):
    """
    Validate the model on the validation set.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to use
        
    Returns:
        tuple: (validation loss, digit accuracy, sequence accuracy)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                             'mps' if torch.backends.mps.is_available() else 
                             'cpu')
    
    model.eval()
    val_loss = 0.0
    correct_digits = 0
    total_digits = 0
    sequence_correct = 0
    
    digit_positions = [f'digit_{i+1}' for i in range(13)]
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Calculate loss
            loss, _ = criterion(outputs, targets)
            val_loss += loss.item()
            
            # Calculate accuracy for individual digits
            all_correct = torch.ones(targets.size(0), dtype=torch.bool).to(device)
            
            for i, pos in enumerate(digit_positions):
                if pos in outputs:
                    _, predicted = torch.max(outputs[pos], 1)
                    digit_target = targets[:, i]
                    correct_digit = (predicted == digit_target)
                    correct_digits += correct_digit.sum().item()
                    all_correct = all_correct & correct_digit
                    total_digits += targets.size(0)
            
            # Calculate full sequence accuracy
            sequence_correct += all_correct.sum().item()
            
    # Calculate metrics
    val_loss = val_loss / len(val_loader)
    digit_acc = 100 * correct_digits / max(total_digits, 1)
    sequence_acc = 100 * sequence_correct / max(len(val_loader.dataset), 1)
    
    return val_loss, digit_acc, sequence_acc

def prepare_data(data_dir, json_file):
    """
    Prepare data for training.
    
    Args:
        data_dir: Path to the directory containing images
        json_file: Path to the JSON file with image paths and labels
    
    Returns:
        tuple: (image_paths, labels)
    """
    # Read JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} items in the dataset")
    
    image_paths = []
    labels = []
    
    for item in data:
        docket_no = str(item['label'])
        if len(docket_no) != 13:
            print(f"Skipping label with incorrect length: {docket_no}")
            continue
        
        image_url = item['image']
        image_path = os.path.join(data_dir, image_url)
        
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(docket_no)
        else:
            print(f"Image not found: {image_path}")
    
    print(f"Found {len(image_paths)} valid labeled images")
    return image_paths, labels

# ============================================================================
# COLAB SETUP AND MAIN FUNCTION
# ============================================================================

def setup_colab():
    """
    Set up the Google Colab environment with necessary packages and configurations.
    """
    try:
        # Mount Google Drive if not already mounted
        # from google.colab import drive
        # drive.mount('/content/drive')
        print("Google Drive mounted successfully.")
        
        # Install required packages if not already installed
        # Using os.system instead of ! magic command
        import os
        os.system('pip install timm tensorboard')
        
        print("Required packages installed.")
        
        # Note: TensorBoard extension loading will be handled in the main function
        # since %load_ext is a Jupyter/Colab magic command that can't be used in regular Python
        
        return True
    except ImportError:
        print("Not running in Google Colab. Skipping Colab setup.")
        return False

def fine_tune_later_digits(model, train_loader, val_loader, num_epochs=20, learning_rate=0.00005, log_dir='runs/fine_tune'):
    """
    Fine-tune a model by freezing the backbone and early digit classifiers,
    focusing only on the later digits (7-13) that need improvement.
    
    Args:
        model: Pretrained DigitSequenceViTModel
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of fine-tuning epochs
        learning_rate: Lower learning rate for fine-tuning
        log_dir: Directory for TensorBoard logs
    
    Returns:
        Tuple of (history, fine-tuned model)
    """
    print("Starting focused fine-tuning of later digits (7-13)...")
    
    # Freeze backbone and early digit classifiers
    for name, param in model.named_parameters():
        # Freeze the ViT backbone
        if 'vit' in name:
            param.requires_grad = False
        
        # Freeze the early digit classifiers (digits 1-6)
        if any(digit in name for digit in ['digit_one', 'digit_two', 'digit_three', 
                                           'digit_four', 'digit_five', 'digit_six']):
            param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
    
    # Train with focused loss on later digits
    history, trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate,  # Lower learning rate for fine-tuning
        scheduler_type='cosine',  
        patience=5,  # Shorter patience for fine-tuning
        log_dir=log_dir
    )
    
    return history, trained_model

def train_with_calibration(model_path=None, train_data_dir=None, val_data_dir=None, 
                          label_smoothing=0.1, focal_gamma=2.0, sequence_penalty=True, 
                          num_epochs=20, learning_rate=0.0001, log_dir='runs/calibrated',
                          checkpoint_every=0):
    """
    Train with confidence calibration techniques.
    
    Args:
        model_path: Path to load/save model
        train_data_dir: Training data directory
        val_data_dir: Validation data directory
        label_smoothing: Amount of label smoothing (0-1)
        focal_gamma: Gamma parameter for focal loss
        sequence_penalty: Whether to use sequence penalty
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        log_dir: Directory for TensorBoard logs
        checkpoint_every: Save checkpoint every N epochs (0 to disable)
        
    Returns:
        Trained model
    """
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 
                         'cpu')
    
    # Print training configuration
    print("=== Training with Confidence Calibration ===")
    print(f"Label Smoothing: {label_smoothing}")
    print(f"Focal Loss Gamma: {focal_gamma}")
    print(f"Sequence Penalty: {sequence_penalty}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Using device: {device}")
    
    # Initialize or load model
    if model_path and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = ResNetDigitSequence()
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Initializing new model")
        model = ResNetDigitSequence()
    
    model = model.to(device)
    
    # Prepare data
    transforms_dict = get_transforms()
    train_transform = transforms_dict['train']
    val_transform = transforms_dict['val']
    
    # Prepare datasets and data loaders
    if train_data_dir and os.path.exists(train_data_dir):
        train_paths, train_labels = prepare_data(
            train_data_dir, 
            os.path.join(train_data_dir, "labels.json")
        )
        train_dataset = DigitSequenceDataset(train_paths, train_labels, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
        print(f"Loaded {len(train_dataset)} training samples")
    else:
        print("No training data directory provided or directory not found")
        return None
    
    if val_data_dir and os.path.exists(val_data_dir):
        val_paths, val_labels = prepare_data(
            val_data_dir,
            os.path.join(val_data_dir, "labels.json")
        )
        val_dataset = DigitSequenceDataset(val_paths, val_labels, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
        print(f"Loaded {len(val_dataset)} validation samples")
    else:
        print("No validation data directory provided or directory not found")
        val_loader = None
    
    # Set up loss function with calibration techniques
    sequence_lambda = 0.1 if sequence_penalty else 0.0
    criterion = DigitSequenceLoss(
        num_digits=13,
        use_sequence_penalty=sequence_penalty,
        sequence_lambda=sequence_lambda,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing
    )
    
    # Use a very small learning rate for fine-tuning to avoid NaN issues
    actual_lr = min(learning_rate, 0.00001)  # Cap at 1e-5 for stability
    print(f"Using learning rate: {actual_lr} (capped for stability)")
    
    # Set up optimizer with smaller weight decay
    optimizer = optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=1e-5)
    
    # Create a custom training function with gradient clipping
    def train_with_grad_clipping(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, checkpoint_every=0, model_prefix='model'):
        """
        Train with gradient clipping and checkpoint saving at regular intervals.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer to use
            criterion: Loss function
            device: Device to use for training
            num_epochs: Number of training epochs
            checkpoint_every: Save checkpoint every N epochs (0 to disable)
            model_prefix: Prefix for saved checkpoint files
            
        Returns:
            tuple: (training history, best model state dict)
        """
        history = {'train_loss': [], 'val_loss': [], 'digit_acc': [], 'sequence_acc': []}
        best_val_loss = float('inf')
        best_model = None
        
        # Set up TensorBoard writer
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            # Training loop with gradient clipping
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                try:
                    loss, _ = criterion(outputs, targets)
                    
                    # Check for NaN loss
                    if torch.isnan(loss).item():
                        print("NaN loss detected, skipping batch")
                        continue
                        
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update weights
                    optimizer.step()
                    
                    # Track loss
                    train_loss += loss.item()
                except Exception as e:
                    print(f"Error in training step: {e}")
                    continue
            
            # Calculate average training loss
            train_loss = train_loss / len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader:
                val_loss, digit_acc, sequence_acc = validate(model, val_loader, criterion, device)
                history['val_loss'].append(val_loss)
                history['digit_acc'].append(digit_acc)
                history['sequence_acc'].append(sequence_acc)
                
                # Log to TensorBoard
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/digit', digit_acc, epoch)
                writer.add_scalar('Accuracy/sequence', sequence_acc, epoch)
                
                # Print summary
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Digit Acc: {digit_acc:.2f}%, Sequence Acc: {sequence_acc:.2f}%")
                
                # Save best model
                if not math.isnan(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict().copy()
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
            
            # Save checkpoint if enabled
            if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
                checkpoint_dir = os.path.join('outputs', 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f'{model_prefix}_checkpoint_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
        
        writer.close()
        return history, best_model if best_model is not None else model.state_dict().copy()
    
    # Run custom training with gradient clipping
    print("Starting training with gradient clipping...")
    _, trained_model_state = train_with_grad_clipping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=num_epochs,
        checkpoint_every=checkpoint_every,
        model_prefix=os.path.basename(model_path).split('.')[0] if model_path else 'model'
    )
    
    # Load best model state
    model.load_state_dict(trained_model_state)
    
    # Save the final model
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, 'calibrated_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Calibrated model saved to {model_save_path}")
    
    return model

def main():
    """Main function to run when script is executed directly."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Digit Sequence Recognition Training')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_path', type=str, help='Path to load/save model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--calibrate', action='store_true', 
                      help='Train with confidence calibration techniques')
    parser.add_argument('--label_smoothing', type=float, default=0.1, 
                      help='Label smoothing amount (0-1)')
    parser.add_argument('--focal_gamma', type=float, default=2.0, 
                      help='Focal loss gamma parameter')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                      help='Learning rate for training')
    parser.add_argument('--checkpoint_every', type=int, default=0,
                      help='Save checkpoint every N epochs (0 to disable)')
    
    args = parser.parse_args()
    
    # Set up directories
    data_dir = args.data_dir if args.data_dir else 'data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'eval')
    test_dir = os.path.join(data_dir, 'test')
    
    # Perform actions based on arguments
    if args.train:
        print("Training model...")
        # Use standard training
        model = train(args.model_path, train_dir, val_dir)
    
    elif args.calibrate:
        print("Training model with confidence calibration techniques...")
        # Use calibrated training
        model = train_with_calibration(
            model_path=args.model_path,
            train_data_dir=train_dir,
            val_data_dir=val_dir,
            label_smoothing=args.label_smoothing,
            focal_gamma=args.focal_gamma,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_every=args.checkpoint_every
        )
    
    elif args.eval:
        print("Evaluating model...")
        # Evaluation code here
        if not args.model_path:
            print("Error: --model_path is required for evaluation")
            sys.exit(1)
        
        # Load model
        model = ResNetDigitSequence()
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        
        # Load test data
        val_transform = get_transforms()['val']
        test_paths, test_labels = prepare_data(
            test_dir,
            os.path.join(test_dir, "labels.json")
        )
        test_dataset = DigitSequenceDataset(test_paths, test_labels, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
        
        # Evaluate model
        print(f"Evaluating model on {len(test_dataset)} test samples...")
        # This is just a placeholder - you should implement a proper evaluation function
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = model(inputs)
                
                # Calculate accuracy
                batch_size = inputs.size(0)
                all_correct = torch.ones(batch_size, dtype=torch.bool).to(DEVICE)
                
                for i in range(13):
                    pos = f'digit_{i+1}'
                    if pos in outputs:
                        _, predicted = torch.max(outputs[pos], 1)
                        correct_mask = (predicted == targets[:, i])
                        all_correct = all_correct & correct_mask
                
                correct += all_correct.sum().item()
                total += batch_size
        
        print(f"Sequence Accuracy: {correct / total * 100:.2f}%")
    
    elif args.visualize:
        print("Visualizing predictions...")
        if not args.model_path:
            print("Error: --model_path is required for visualization")
            sys.exit(1)
            
        # Load model
        model = ResNetDigitSequence()
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        
        # Load some test data
        val_transform = get_transforms()['val']
        test_paths, test_labels = prepare_data(
            test_dir,
            os.path.join(test_dir, "labels.json")
        )
        
        # Visualize a few samples
        os.makedirs('visualizations', exist_ok=True)
        for i in range(min(5, len(test_paths))):
            visualize_sample(
                test_paths[i], 
                test_labels[i], 
                save_dir='visualizations',
                model=model,
                transform=val_transform
            )
        
        print("Visualizations saved to 'visualizations' directory")
    
    else:
        print("No action specified. Use --train, --eval, --calibrate, or --visualize.")
        parser.print_help()

if __name__ == "__main__":
    main() 