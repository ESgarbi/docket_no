#!/usr/bin/env python3
"""
Evaluation script for the Docket Number Sequence Recognition Model.
Runs comprehensive evaluation on test data and generates a detailed report.
"""

import os
import json
import argparse
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import random
from pathlib import Path

# Import model and dataset classes
from google_colab_training_resnet import ResNetDigitSequence, DigitSequenceDataset, prepare_data
from temperature_scaling import DigitTemperatureScaler

# Set up constants
NUM_DIGITS = 13
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                     'mps' if torch.backends.mps.is_available() else 
                     'cpu')
TEST_DIR = "/Users/erick/git/prod/docket_no/sequence/001/test"
# We'll use the test directory for calibration as well
CALIBRATION_DATA_RATIO = 0.2  # Use 20% of test data for calibration
MODEL_PATH = "/Users/erick/git/prod/docket_no/outputs/final_model.pth"
TEMP_SCALER_PATH = "/Users/erick/git/prod/docket_no/outputs/temperature_scaler.pth"
REPORTS_DIR = "/Users/erick/git/prod/docket_no/evaluation_reports"


def load_model(model_path):
    """Load the trained model from the specified path."""
    model = ResNetDigitSequence()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def get_val_transform():
    """Get the validation/test transform for the images."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def evaluate_model(model, test_loader, temp_scaler=None):
    """Run evaluation on the model and return evaluation metrics."""
    results = {
        'total_samples': 0,
        'digit_correct': 0,
        'digit_total': 0,
        'sequence_correct': 0,
        'per_digit_correct': [0] * NUM_DIGITS,
        'per_digit_total': [0] * NUM_DIGITS,
        'confusion_matrices': [np.zeros((10, 10), dtype=int) for _ in range(NUM_DIGITS)],
        'all_predictions': [],
        'all_targets': [],
        'sample_results': [],  # Store sample images and predictions for visualization
        'wrong_predictions_confidence': [],  # Track confidence scores for wrong predictions
        'wrong_predictions_calibrated_confidence': [],  # Track calibrated confidence scores
        'wrong_sequences': []  # Store complete sequences that contain at least one error
    }
    
    digit_positions = [f'digit_{i+1}' for i in range(NUM_DIGITS)]
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            results['total_samples'] += inputs.size(0)
            
            # Get model predictions
            outputs = model(inputs)
            
            # Apply temperature scaling if provided
            calibrated_probs = None
            if temp_scaler is not None:
                calibrated_probs = temp_scaler.calibrate_outputs(outputs)
            
            # Store random samples for visualization (every 10th batch)
            if batch_idx % 10 == 0 and batch_idx < 100:
                for i in range(min(2, inputs.size(0))):  # Store up to 2 samples per selected batch
                    sample_idx = i
                    sample_input = inputs[sample_idx].cpu()
                    sample_target = targets[sample_idx].cpu().numpy()
                    sample_predictions = {}
                    sample_confidences = {}
                    sample_calibrated_confidences = {}
                    
                    for pos_idx, pos in enumerate(digit_positions):
                        if pos in outputs:
                            # Original confidence
                            logits = outputs[pos][sample_idx]
                            probs = F.softmax(logits, dim=0).cpu().numpy()
                            pred = np.argmax(probs)
                            sample_predictions[pos] = int(pred)
                            sample_confidences[pos] = float(probs[pred])
                            
                            # Calibrated confidence if available
                            if calibrated_probs is not None and pos in calibrated_probs:
                                cal_probs = calibrated_probs[pos][sample_idx].cpu().numpy()
                                sample_calibrated_confidences[pos] = float(cal_probs[pred])
                    
                    sample_result = {
                        'input': sample_input,
                        'target': sample_target,
                        'predictions': sample_predictions,
                        'confidences': sample_confidences,
                        'calibrated_confidences': sample_calibrated_confidences if calibrated_probs else None
                    }
                    results['sample_results'].append(sample_result)
            
            # Calculate accuracy for individual digits
            all_correct = torch.ones(inputs.size(0), dtype=torch.bool).to(DEVICE)
            predictions = []
            
            for i, pos in enumerate(digit_positions):
                if pos in outputs:
                    logits = outputs[pos]
                    probs = F.softmax(logits, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                    digit_target = targets[:, i]
                    
                    # Get calibrated confidence if available
                    calibrated_confidence = None
                    if calibrated_probs is not None and pos in calibrated_probs:
                        _, calibrated_confidence = torch.max(calibrated_probs[pos], 1)
                    
                    # Update per-digit statistics
                    correct_mask = (predicted == digit_target)
                    results['per_digit_correct'][i] += correct_mask.sum().item()
                    results['per_digit_total'][i] += targets.size(0)
                    results['digit_correct'] += correct_mask.sum().item()
                    results['digit_total'] += targets.size(0)
                    
                    # Collect wrong predictions with confidence scores
                    wrong_indices = torch.where(~correct_mask)[0]
                    for idx in wrong_indices:
                        wrong_pred = {
                            'position': i + 1,
                            'true_value': digit_target[idx].item(),
                            'predicted_value': predicted[idx].item(),
                            'confidence': confidence[idx].item(),
                            'image_idx': batch_idx * inputs.size(0) + idx.item()
                        }
                        results['wrong_predictions_confidence'].append(wrong_pred)
                        
                        # Add calibrated confidence if available
                        if calibrated_confidence is not None:
                            cal_wrong_pred = wrong_pred.copy()
                            cal_wrong_pred['confidence'] = calibrated_confidence[idx].item()
                            results['wrong_predictions_calibrated_confidence'].append(cal_wrong_pred)
                    
                    # Update confusion matrix
                    for t, p in zip(digit_target.cpu().numpy(), predicted.cpu().numpy()):
                        results['confusion_matrices'][i][t, p] += 1
                    
                    # Track full sequence correctness
                    all_correct = all_correct & correct_mask
                    
                    # Collect predictions for this batch
                    predictions.append(predicted.cpu().numpy())
            
            # Calculate full sequence accuracy
            results['sequence_correct'] += all_correct.sum().item()
            
            # Store predictions and targets for later analysis
            if len(predictions) == NUM_DIGITS:
                batch_predictions = np.stack(predictions, axis=1)
                batch_targets = targets.cpu().numpy()
                results['all_predictions'].extend(batch_predictions.tolist())
                results['all_targets'].extend(batch_targets.tolist())
                
                # Find incorrect sequences and store them
                wrong_indices = torch.where(~all_correct)[0].cpu().numpy()
                for idx in wrong_indices:
                    wrong_sequence = {
                        'image_idx': batch_idx * inputs.size(0) + idx,
                        'predicted': [int(pred) for pred in batch_predictions[idx]],
                        'ground_truth': [int(target) for target in batch_targets[idx]]
                    }
                    results['wrong_sequences'].append(wrong_sequence)
    
    # Calculate final metrics
    results['digit_accuracy'] = results['digit_correct'] / max(results['digit_total'], 1) * 100
    results['sequence_accuracy'] = results['sequence_correct'] / max(results['total_samples'], 1) * 100
    results['per_digit_accuracy'] = [
        correct / max(total, 1) * 100 
        for correct, total in zip(results['per_digit_correct'], results['per_digit_total'])
    ]
    
    return results


def visualize_digit_accuracy(accuracies, output_path):
    """Visualize per-digit accuracy as a bar chart."""
    plt.figure(figsize=(14, 6))
    plt.bar(range(1, NUM_DIGITS + 1), accuracies, color='royalblue')
    plt.xlabel('Digit Position')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Digit Accuracy')
    plt.xticks(range(1, NUM_DIGITS + 1))
    plt.ylim(min(80, min(accuracies) - 5), 100.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i + 1, acc + 0.5, f'{acc:.2f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def visualize_confusion_matrices(confusion_matrices, output_dir):
    """Visualize confusion matrices for each digit position."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, cm in enumerate(confusion_matrices):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Digit Position {i+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_digit_{i+1}.png'))
        plt.close()


def visualize_sample_predictions(samples, output_dir):
    """Visualize sample predictions with input images."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Include some wrong predictions with high confidence if available
    for i, sample in enumerate(random.sample(samples, min(10, len(samples)))):
        # Convert tensor to numpy image
        img = sample['input'].permute(1, 2, 0).numpy()
        img = np.clip((img * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(img)
        ax.axis('off')
        
        # Create prediction string and target string
        target_str = ''.join([str(d) for d in sample['target']])
        
        pred_digits = []
        conf_values = []
        cal_conf_values = []
        correct_flags = []
        
        for j in range(NUM_DIGITS):
            pos = f'digit_{j+1}'
            if pos in sample['predictions']:
                predicted = sample['predictions'][pos]
                pred_digits.append(str(predicted))
                conf_values.append(sample['confidences'][pos])
                if sample['calibrated_confidences'] and pos in sample['calibrated_confidences']:
                    cal_conf_values.append(sample['calibrated_confidences'][pos])
                else:
                    cal_conf_values.append(None)
                correct_flags.append(predicted == sample['target'][j])
            else:
                pred_digits.append('?')
                conf_values.append(0.0)
                cal_conf_values.append(None)
                correct_flags.append(False)
        
        pred_str = ''.join(pred_digits)
        
        # Add prediction information as title
        match = pred_str == target_str
        color = 'green' if match else 'red'
        
        fig.suptitle(f"Target: {target_str}\nPrediction: {pred_str}", 
                     color=color, fontsize=14, fontweight='bold')
        
        # Adjust layout to fit both confidence bars
        plt.subplots_adjust(bottom=0.40)
        
        # Original confidence bars
        confidence_ax = fig.add_axes([0.1, 0.25, 0.8, 0.1])
        pos = np.arange(NUM_DIGITS)
        colors = ['green' if flag else 'red' for flag in correct_flags]
        
        bars = confidence_ax.bar(pos, conf_values, color=colors)
        confidence_ax.set_xlim(-0.5, NUM_DIGITS - 0.5)
        confidence_ax.set_ylim(0, 1.05)
        confidence_ax.set_xticks(pos)
        confidence_ax.set_xticklabels([f'{i+1}' for i in range(NUM_DIGITS)])
        confidence_ax.set_xlabel('Digit Position')
        confidence_ax.set_ylabel('Confidence')
        confidence_ax.set_title('Original Confidence')
        confidence_ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add confidence values on top of the bars
        for j, (bar, conf, is_correct) in enumerate(zip(bars, conf_values, correct_flags)):
            confidence_ax.text(j, conf + 0.02, f"{conf:.2f}", ha='center', 
                             va='bottom', color='black', fontsize=8, fontweight='bold')
        
        # Calibrated confidence bars (if available)
        if any(c is not None for c in cal_conf_values):
            cal_confidence_ax = fig.add_axes([0.1, 0.1, 0.8, 0.1])
            cal_bars = cal_confidence_ax.bar(pos, 
                                            [c if c is not None else 0 for c in cal_conf_values], 
                                            color=colors)
            cal_confidence_ax.set_xlim(-0.5, NUM_DIGITS - 0.5)
            cal_confidence_ax.set_ylim(0, 1.05)
            cal_confidence_ax.set_xticks(pos)
            cal_confidence_ax.set_xticklabels([f'{i+1}' for i in range(NUM_DIGITS)])
            cal_confidence_ax.set_xlabel('Digit Position')
            cal_confidence_ax.set_ylabel('Confidence')
            cal_confidence_ax.set_title('Calibrated Confidence')
            cal_confidence_ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Add calibrated confidence values on top of the bars
            for j, (bar, conf, is_correct) in enumerate(zip(cal_bars, cal_conf_values, correct_flags)):
                if conf is not None:
                    cal_confidence_ax.text(j, conf + 0.02, f"{conf:.2f}", ha='center', 
                                         va='bottom', color='black', fontsize=8, fontweight='bold')
        
        plt.savefig(os.path.join(output_dir, f'sample_prediction_{i+1}.png'))
        plt.close()
    
    
def visualize_high_confidence_errors(wrong_predictions, output_dir, samples=5):
    """Visualize predictions with high confidence but wrong answers."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort wrong predictions by confidence (highest first)
    sorted_errors = sorted(wrong_predictions, key=lambda x: x['confidence'], reverse=True)
    
    # Create a visualization of high confidence errors
    plt.figure(figsize=(12, 8))
    
    top_errors = sorted_errors[:min(samples, len(sorted_errors))]
    positions = [f"Pos {p['position']}" for p in top_errors]
    confidences = [p['confidence'] for p in top_errors]
    labels = [f"{p['true_value']}→{p['predicted_value']}" for p in top_errors]
    
    bars = plt.barh(positions, confidences, color='firebrick')
    plt.xlim(0, 1.05)
    plt.xlabel('Confidence Score')
    plt.title('Top High-Confidence Errors')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Add labels to bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                labels[i], va='center', fontweight='bold')
        plt.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height()/2, 
                f"{confidences[i]:.2f}", va='center', ha='right', 
                color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'high_confidence_errors.png'))
    plt.close()


def analyze_wrong_predictions_confidence(wrong_predictions, output_path):
    """Analyze confidence scores of wrong predictions."""
    if not wrong_predictions:
        return {}
    
    # Group wrong predictions by confidence ranges
    confidence_ranges = [(0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0)]
    confidence_counts = {f"{int(low*100)}-{int(high*100)}%": 0 for low, high in confidence_ranges}
    
    # Count wrong predictions in each confidence range
    for wrong_pred in wrong_predictions:
        conf = wrong_pred['confidence']
        for (low, high), range_key in zip(confidence_ranges, confidence_counts.keys()):
            if low <= conf < high:
                confidence_counts[range_key] += 1
                break
                
    # Get per-position confidence data
    position_data = {}
    for pos in range(1, NUM_DIGITS + 1):
        position_preds = [p for p in wrong_predictions if p['position'] == pos]
        if position_preds:
            confidences = [p['confidence'] for p in position_preds]
            position_data[pos] = {
                'count': len(position_preds),
                'avg_confidence': sum(confidences) / len(confidences),
                'high_confidence': sum(1 for c in confidences if c > 0.9),
                'max_confidence': max(confidences) if confidences else 0
            }
    
    # Visualize confidence distribution of wrong predictions
    plt.figure(figsize=(12, 6))
    bars = plt.bar(confidence_counts.keys(), confidence_counts.values(), color='firebrick')
    plt.xlabel('Confidence Range')
    plt.ylabel('Number of Wrong Predictions')
    plt.title('Confidence Distribution of Wrong Predictions')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Calculate aggregate statistics
    all_confidences = [p['confidence'] for p in wrong_predictions]
    stats = {
        'total_wrong': len(wrong_predictions),
        'high_confidence_errors': sum(1 for c in all_confidences if c > 0.9),
        'avg_confidence': sum(all_confidences) / max(len(all_confidences), 1),
        'confidence_distribution': confidence_counts,
        'position_data': position_data
    }
    
    return stats


def compare_confidence_distributions(original, calibrated, output_path):
    """
    Compare the confidence distributions before and after calibration.
    """
    if not original or not calibrated:
        return
    
    # Group predictions by confidence ranges
    confidence_ranges = [(0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0)]
    range_labels = [f"{int(low*100)}-{int(high*100)}%" for low, high in confidence_ranges]
    
    # Count wrong predictions in each confidence range
    orig_counts = [0] * len(confidence_ranges)
    cal_counts = [0] * len(confidence_ranges)
    
    for pred in original:
        conf = pred['confidence']
        for i, (low, high) in enumerate(confidence_ranges):
            if low <= conf < high:
                orig_counts[i] += 1
                break
    
    for pred in calibrated:
        conf = pred['confidence']
        for i, (low, high) in enumerate(confidence_ranges):
            if low <= conf < high:
                cal_counts[i] += 1
                break
    
    # Create the comparison plot
    plt.figure(figsize=(14, 6))
    
    x = np.arange(len(range_labels))
    width = 0.35
    
    # Plot bars side by side
    plt.bar(x - width/2, orig_counts, width, label='Original', color='firebrick')
    plt.bar(x + width/2, cal_counts, width, label='Calibrated', color='royalblue')
    
    plt.xlabel('Confidence Range')
    plt.ylabel('Number of Wrong Predictions')
    plt.title('Confidence Distribution Before and After Calibration')
    plt.xticks(x, range_labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count labels
    for i, count in enumerate(orig_counts):
        if count > 0:
            plt.text(i - width/2, count + 0.5, str(count), ha='center', va='bottom', color='black')
    
    for i, count in enumerate(cal_counts):
        if count > 0:
            plt.text(i + width/2, count + 0.5, str(count), ha='center', va='bottom', color='black')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Return statistics for reporting
    orig_high_conf = sum(orig_counts[4:])  # >90% confidence
    cal_high_conf = sum(cal_counts[4:])    # >90% confidence
    
    stats = {
        'original': {
            'distribution': dict(zip(range_labels, orig_counts)),
            'high_confidence_errors': orig_high_conf,
            'total': sum(orig_counts)
        },
        'calibrated': {
            'distribution': dict(zip(range_labels, cal_counts)),
            'high_confidence_errors': cal_high_conf,
            'total': sum(cal_counts)
        }
    }
    
    return stats


def create_misses_csv(wrong_predictions, wrong_sequences, output_path):
    """
    Create a CSV file containing all misses with predicted, ground_truth, and DocumentVersionID columns.
    Each row contains the full sequence, not individual digit predictions.
    
    Args:
        wrong_predictions: List of dictionaries containing wrong prediction data
        wrong_sequences: List of dictionaries containing full sequences with errors
        output_path: Path to save the CSV file
    
    Returns:
        DataFrame containing the miss data
    """
    if not wrong_sequences:
        return None
    
    # Create a DataFrame with full sequences
    data = []
    
    for seq in wrong_sequences:
        data.append({
            'predicted': ''.join([str(digit) for digit in seq['predicted']]),
            'ground_truth': ''.join([str(digit) for digit in seq['ground_truth']]),
            'DocumentVersionID': seq['image_idx']
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df


def visualize_position_failures(wrong_predictions, output_path):
    """
    Create a bar chart showing failures by position (0-13).
    
    Args:
        wrong_predictions: List of dictionaries containing wrong prediction data
        output_path: Path to save the visualization
    """
    if not wrong_predictions:
        return
    
    # Count failures by position
    position_counts = {}
    for pred in wrong_predictions:
        pos = pred['position']
        position_counts[pos] = position_counts.get(pos, 0) + 1
    
    positions = sorted(position_counts.keys())
    counts = [position_counts.get(pos, 0) for pos in positions]
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.bar(positions, counts, color='firebrick')
    plt.xlabel('Position (1-13)')
    plt.ylabel('Number of Failures')
    plt.title('Failures by Position')
    plt.xticks(positions)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count labels on top of bars
    for i, count in enumerate(counts):
        plt.text(positions[i], count + 0.5, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_report(results, report_dir, temp_scaler=None):
    """Generate a comprehensive evaluation report."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    report_dir = os.path.join(report_dir, f'evaluation_{timestamp}')
    os.makedirs(report_dir, exist_ok=True)
    
    # Create subdirectories for visualizations
    cm_dir = os.path.join(report_dir, 'confusion_matrices')
    samples_dir = os.path.join(report_dir, 'sample_predictions')
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    # Generate visualizations
    visualize_digit_accuracy(results['per_digit_accuracy'], 
                           os.path.join(report_dir, 'per_digit_accuracy.png'))
    visualize_confusion_matrices(results['confusion_matrices'], cm_dir)
    visualize_sample_predictions(results['sample_results'], samples_dir)
    
    # Analyze and visualize wrong predictions confidence
    wrong_confidence_stats = {}
    calibration_comparison_stats = {}
    misses_df = None
    
    if results['wrong_predictions_confidence']:
        wrong_confidence_stats = analyze_wrong_predictions_confidence(
            results['wrong_predictions_confidence'],
            os.path.join(report_dir, 'wrong_predictions_confidence.png')
        )
        visualize_high_confidence_errors(
            results['wrong_predictions_confidence'],
            report_dir
        )
        
        # Create CSV with all misses
        misses_csv_path = os.path.join(report_dir, 'misses.csv')
        misses_df = create_misses_csv(
            results['wrong_predictions_confidence'],
            results['wrong_sequences'],
            misses_csv_path
        )
        
        # Create visualization of failures by position
        visualize_position_failures(
            results['wrong_predictions_confidence'],
            os.path.join(report_dir, 'position_failures.png')
        )
        
        # Compare original vs calibrated confidence if available
        if results['wrong_predictions_calibrated_confidence']:
            calibration_comparison_stats = compare_confidence_distributions(
                results['wrong_predictions_confidence'],
                results['wrong_predictions_calibrated_confidence'],
                os.path.join(report_dir, 'confidence_calibration_comparison.png')
            )
    
    # Generate text report (markdown)
    report_path = os.path.join(report_dir, 'evaluation_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Docket Number Sequence Recognition - Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall metrics
        f.write("## Overall Metrics\n\n")
        f.write(f"- **Test Samples:** {results['total_samples']}\n")
        f.write(f"- **Per-Digit Accuracy:** {results['digit_accuracy']:.2f}%\n")
        f.write(f"- **Full Sequence Accuracy:** {results['sequence_accuracy']:.2f}%\n\n")
        
        # Add misses table and visualization
        if misses_df is not None:
            f.write("## Prediction Misses\n\n")
            
            # Position failures visualization
            f.write("### Failures by Position\n\n")
            f.write("![Position Failures](position_failures.png)\n\n")
            
            # Misses table (show top 10)
            f.write("### Sample of Prediction Misses\n\n")
            f.write("The following table shows a sample of prediction misses. The full list is available in `misses.csv`.\n\n")
            f.write("| ID | Position | Ground Truth | Predicted | Confidence |\n")
            f.write("|------|----------|--------------|-----------|------------|\n")
            
            # Display top 10 misses
            sample_rows = min(10, len(misses_df))
            for _, row in misses_df.head(sample_rows).iterrows():
                f.write(f"| {row['id']} | {row['position']} | {row['ground_truth']} | {row['predicted']} | {row['confidence']:.2f} |\n")
            
            f.write(f"\nTotal misses: {len(misses_df)} (Full data available in `misses.csv`)\n\n")
        
        # Temperature scaling information if available
        if temp_scaler and temp_scaler.optimized:
            f.write("## Temperature Scaling Parameters\n\n")
            f.write("The following temperature values were used to calibrate confidence scores:\n\n")
            f.write("| Digit Position | Temperature Value |\n")
            f.write("|----------------|-------------------|\n")
            
            for i in range(NUM_DIGITS):
                f.write(f"| {i+1} | {temp_scaler.temperatures[i].item():.4f} |\n")
            
            f.write("\nHigher values (>1.0) indicate the model was overconfident for that position and confidence scores have been reduced.\n\n")
        
        # Per-digit accuracy table
        f.write("## Per-Digit Accuracy\n\n")
        f.write("| Digit Position | Accuracy (%) | Correct | Total |\n")
        f.write("|----------------|-------------|---------|-------|\n")
        
        for i in range(NUM_DIGITS):
            f.write(f"| {i+1} | {results['per_digit_accuracy'][i]:.2f} | {results['per_digit_correct'][i]} | {results['per_digit_total'][i]} |\n")
        
        # Wrong prediction confidence analysis
        if wrong_confidence_stats:
            f.write("\n## Wrong Prediction Confidence Analysis\n\n")
            f.write(f"- **Total Wrong Predictions:** {wrong_confidence_stats['total_wrong']}\n")
            f.write(f"- **Average Confidence of Wrong Predictions:** {wrong_confidence_stats['avg_confidence']:.2f}\n")
            f.write(f"- **High-Confidence Errors (>90%):** {wrong_confidence_stats['high_confidence_errors']} ({wrong_confidence_stats['high_confidence_errors']/max(wrong_confidence_stats['total_wrong'], 1)*100:.2f}%)\n\n")
            
            f.write("### Confidence Distribution of Wrong Predictions\n\n")
            f.write("| Confidence Range | Count | Percentage |\n")
            f.write("|------------------|-------|------------|\n")
            
            for conf_range, count in wrong_confidence_stats['confidence_distribution'].items():
                percentage = count / max(wrong_confidence_stats['total_wrong'], 1) * 100
                f.write(f"| {conf_range} | {count} | {percentage:.2f}% |\n")
            
            # Add confidence calibration comparison if available
            if calibration_comparison_stats:
                f.write("\n### Confidence Calibration Effect\n\n")
                f.write("The table below shows how temperature scaling affects the distribution of confidence scores for wrong predictions:\n\n")
                f.write("| Confidence Range | Original Count | Calibrated Count | Change |\n")
                f.write("|------------------|---------------|-----------------|--------|\n")
                
                orig_dist = calibration_comparison_stats['original']['distribution']
                cal_dist = calibration_comparison_stats['calibrated']['distribution']
                
                for conf_range in orig_dist.keys():
                    orig_count = orig_dist[conf_range]
                    cal_count = cal_dist[conf_range]
                    change = cal_count - orig_count
                    change_str = f"{change:+d}" if change != 0 else "0"
                    f.write(f"| {conf_range} | {orig_count} | {cal_count} | {change_str} |\n")
                
                # Summary of calibration effect
                orig_high = calibration_comparison_stats['original']['high_confidence_errors']
                cal_high = calibration_comparison_stats['calibrated']['high_confidence_errors']
                high_change = cal_high - orig_high
                high_pct_change = (high_change / max(orig_high, 1)) * 100
                
                f.write(f"\nHigh-confidence errors (>90%): **{orig_high}** → **{cal_high}** ")
                if high_change < 0:
                    f.write(f"({high_change:d}, {high_pct_change:.1f}% reduction)\n")
                else:
                    f.write(f"({high_change:+d}, {high_pct_change:.1f}% increase)\n")
                
                f.write("\n![Confidence Calibration Comparison](confidence_calibration_comparison.png)\n\n")
            
            f.write("\n### Problematic Positions with High-Confidence Errors\n\n")
            f.write("| Position | Wrong Count | Avg. Confidence | High-Conf Errors (>90%) | Max Confidence |\n")
            f.write("|----------|-------------|-----------------|--------------------------|---------------|\n")
            
            # Sort positions by number of high-confidence errors
            sorted_positions = sorted(
                wrong_confidence_stats['position_data'].items(),
                key=lambda x: x[1]['high_confidence'],
                reverse=True
            )
            
            for pos, data in sorted_positions:
                f.write(f"| {pos} | {data['count']} | {data['avg_confidence']:.2f} | {data['high_confidence']} | {data['max_confidence']:.2f} |\n")
                
            f.write("\n![Confidence Distribution](wrong_predictions_confidence.png)\n\n")
            f.write("![High Confidence Errors](high_confidence_errors.png)\n\n")
        
        # Common error patterns
        if len(results['all_predictions']) > 0:
            f.write("\n## Error Analysis\n\n")
            
            # Find sequences with errors
            error_counts = {i+1: 0 for i in range(NUM_DIGITS)}
            error_examples = {i+1: [] for i in range(NUM_DIGITS)}
            
            for pred, target in zip(results['all_predictions'], results['all_targets']):
                for i in range(NUM_DIGITS):
                    if pred[i] != target[i]:
                        error_counts[i+1] += 1
                        if len(error_examples[i+1]) < 5:  # limit to 5 examples per position
                            error_examples[i+1].append((target[i], pred[i]))
            
            # Most common error positions
            f.write("### Most Challenging Digit Positions\n\n")
            f.write("| Rank | Position | Error Count | Error Rate (%) | Common Errors |\n")
            f.write("|------|----------|-------------|----------------|---------------|\n")
            
            sorted_positions = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
            for rank, (pos, count) in enumerate(sorted_positions[:5], 1):
                if count > 0:
                    error_rate = count / results['per_digit_total'][pos-1] * 100
                    examples = [f"{t}→{p}" for t, p in error_examples[pos][:3]]
                    example_str = ", ".join(examples)
                    f.write(f"| {rank} | {pos} | {count} | {error_rate:.2f} | {example_str} |\n")
        
        # Visualization references
        f.write("\n## Visualizations\n\n")
        f.write("### Per-Digit Accuracy\n\n")
        f.write("![Per-Digit Accuracy](per_digit_accuracy.png)\n\n")
        
        f.write("### Sample Predictions\n\n")
        f.write("See the `sample_predictions` directory for visualizations of sample predictions.\n\n")
        
        f.write("### Confusion Matrices\n\n")
        f.write("See the `confusion_matrices` directory for confusion matrices for each digit position.\n\n")
        
        # Generate recommendations
        f.write("## Recommendations\n\n")
        
        # Find most challenging positions
        challenging_positions = [i+1 for i, acc in enumerate(results['per_digit_accuracy']) 
                                if acc < (sum(results['per_digit_accuracy']) / NUM_DIGITS) - 2]
        
        if challenging_positions:
            pos_str = ", ".join([str(p) for p in challenging_positions])
            f.write(f"- **Focus on improving digit positions {pos_str}**, which have below-average accuracy.\n")
        
        # Check for sequence-level vs. digit-level performance gap
        if results['sequence_accuracy'] < results['digit_accuracy'] * 0.9:
            gap = results['digit_accuracy'] - results['sequence_accuracy']
            f.write(f"- **Strengthen sequence consistency**: There's a {gap:.2f}% gap between digit and sequence accuracy.\n")
        
        if results['sequence_accuracy'] < 95:
            f.write("- **Consider additional data augmentation** to improve robustness for challenging cases.\n")
            
        # Recommendations based on confidence analysis
        if wrong_confidence_stats and wrong_confidence_stats['high_confidence_errors'] > 0:
            high_conf_pct = wrong_confidence_stats['high_confidence_errors']/max(wrong_confidence_stats['total_wrong'], 1)*100
            
            if not temp_scaler or not temp_scaler.optimized:
                f.write("- **Implement temperature scaling** to calibrate confidence scores and reduce overconfidence.\n")
            elif high_conf_pct > 20 and calibration_comparison_stats and \
                 calibration_comparison_stats['calibrated']['high_confidence_errors'] > \
                 calibration_comparison_stats['original']['high_confidence_errors'] * 0.7:
                # If calibration didn't reduce high-confidence errors enough
                f.write("- **Fine-tune temperature scaling parameters** to further reduce overconfidence.\n")
                f.write("- **Consider more advanced calibration techniques** beyond temperature scaling.\n")
            
            # Identify positions with high-confidence errors
            high_conf_positions = [pos for pos, data in wrong_confidence_stats['position_data'].items() 
                                if data['high_confidence'] > 0]
            if high_conf_positions:
                pos_str = ", ".join([str(p) for p in high_conf_positions[:3]])
                f.write(f"- **Add more diverse training examples for positions {pos_str}** to reduce high-confidence errors.\n")
                
                # If the last positions have most high-confidence errors
                if all(p > NUM_DIGITS/2 for p in high_conf_positions[:3]):
                    f.write("- **Consider a specialized model for the second half of the sequence** where most errors occur.\n")
        
        # Report creation summary
        f.write(f"\n\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Generate summary CSV for tracking over time
    summary_data = {
        'Date': [datetime.now().strftime('%Y-%m-%d')],
        'Test Samples': [results['total_samples']],
        'Per-Digit Accuracy': [results['digit_accuracy']],
        'Sequence Accuracy': [results['sequence_accuracy']]
    }
    
    # Add per-digit accuracies
    for i in range(NUM_DIGITS):
        summary_data[f'Digit_{i+1}_Accuracy'] = [results['per_digit_accuracy'][i]]
    
    # Add wrong prediction confidence metrics
    if wrong_confidence_stats:
        summary_data['Wrong_Predictions'] = [wrong_confidence_stats['total_wrong']]
        summary_data['Avg_Wrong_Confidence'] = [wrong_confidence_stats['avg_confidence']]
        summary_data['High_Confidence_Errors'] = [wrong_confidence_stats['high_confidence_errors']]
        
        # Add calibration metrics if available
        if calibration_comparison_stats:
            summary_data['Cal_High_Confidence_Errors'] = [calibration_comparison_stats['calibrated']['high_confidence_errors']]
            summary_data['Confidence_Reduction_Pct'] = [
                (1 - calibration_comparison_stats['calibrated']['high_confidence_errors'] / 
                 max(calibration_comparison_stats['original']['high_confidence_errors'], 1)) * 100
            ]
    
    df = pd.DataFrame(summary_data)
    summary_path = os.path.join(report_dir, 'evaluation_summary.csv')
    df.to_csv(summary_path, index=False)
    
    # Copy the summary to the main evaluation_summaries.csv file if it exists
    main_summary_path = os.path.join(REPORTS_DIR, 'evaluation_summaries.csv')
    if os.path.exists(main_summary_path):
        main_df = pd.read_csv(main_summary_path)
        main_df = pd.concat([main_df, df], ignore_index=True)
        main_df.to_csv(main_summary_path, index=False)
    else:
        df.to_csv(main_summary_path, index=False)
    
    return report_dir


def generate_evaluation_table_for_readme(results, report_dir):
    """Generate a markdown table entry for the README file."""
    date = datetime.now().strftime('%Y-%m-%d')
    digit_acc = f"{results['digit_accuracy']:.2f}%"
    seq_acc = f"{results['sequence_accuracy']:.2f}%"
    
    relative_path = os.path.relpath(report_dir, start="/Users/erick/git/prod/docket_no")
    
    return f"| {date} | Test ({results['total_samples']} samples) | {digit_acc} | {seq_acc} | [Report]({relative_path}/evaluation_report.md) |"


def create_temperature_scaler(model, test_data, test_labels, force_recalibrate=False):
    """Create and load (or create) a temperature scaler using a portion of test data."""
    # Check if temperature scaler already exists
    if os.path.exists(TEMP_SCALER_PATH) and not force_recalibrate:
        print(f"Loading temperature scaler from {TEMP_SCALER_PATH}")
        temp_scaler = DigitTemperatureScaler(model, num_digits=NUM_DIGITS, device=DEVICE)
        temp_scaler.load(TEMP_SCALER_PATH)
        return temp_scaler
    
    # Create a new temperature scaler
    print("Creating and optimizing new temperature scaler...")
    
    if not test_data or len(test_data) == 0:
        print("No data found for calibration. Skipping temperature scaling.")
        return None
    
    # Use a small subset of test data for calibration
    # Shuffle the paths and labels together
    combined = list(zip(test_data, test_labels))
    random.shuffle(combined)
    
    # Take the first portion for calibration
    calibration_size = int(len(test_data) * CALIBRATION_DATA_RATIO)
    calibration_data = combined[:calibration_size]
    
    # Unzip the calibration data
    calibration_paths, calibration_labels = zip(*calibration_data) if calibration_data else ([], [])
    
    print(f"Using {len(calibration_paths)} samples for temperature calibration")
    
    val_transform = get_val_transform()
    calibration_dataset = DigitSequenceDataset(calibration_paths, calibration_labels, transform=val_transform)
    calibration_loader = DataLoader(calibration_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Create and optimize temperature scaler
    temp_scaler = DigitTemperatureScaler(model, num_digits=NUM_DIGITS, device=DEVICE)
    temperature_values = temp_scaler.optimize_temperatures(calibration_loader)
    
    # Save the temperature scaler
    os.makedirs(os.path.dirname(TEMP_SCALER_PATH), exist_ok=True)
    temp_scaler.save(TEMP_SCALER_PATH)
    
    return temp_scaler


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Docket Number Sequence Recognition Model")
    parser.add_argument("--force_recalibrate", action="store_true", help="Force recalibration of temperature scaling")
    parser.add_argument("--no_calibration", action="store_true", help="Disable temperature scaling calibration")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the model to evaluate")
    args = parser.parse_args()
    
    # Create report directory if it doesn't exist
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)
    
    # Prepare the test data
    print(f"Loading test data from {TEST_DIR}")
    test_paths, test_labels = prepare_data(TEST_DIR, os.path.join(TEST_DIR, "labels.json"))
    test_paths_extended, test_labels_extended = prepare_data('/Users/erick/git/prod/docket_no/2005', os.path.join('/Users/erick/git/prod/docket_no/2005', "labels.json"))

    # merge the two lists
    test_paths = test_paths + test_paths_extended
    test_labels = test_labels + test_labels_extended

    # Create or load temperature scaler
    temp_scaler = None
    if not args.no_calibration:
        temp_scaler = create_temperature_scaler(model, test_paths, test_labels, force_recalibrate=args.force_recalibrate)
    
    # For evaluation, we'll use all the data if no calibration, 
    # or the remaining data if we did calibration
    if temp_scaler and not args.force_recalibrate and not os.path.exists(TEMP_SCALER_PATH):
        # If we just created a new temperature scaler, remove the calibration data from test set
        combined = list(zip(test_paths, test_labels))
        random.shuffle(combined)
        
        # Remove the calibration portion
        calibration_size = int(len(test_paths) * CALIBRATION_DATA_RATIO)
        evaluation_data = combined[calibration_size:]
        
        # Unzip the evaluation data
        eval_paths, eval_labels = zip(*evaluation_data) if evaluation_data else (test_paths, test_labels)
        
        print(f"Using {len(eval_paths)} samples for evaluation (excluding calibration data)")
    else:
        # Use all test data for evaluation
        eval_paths, eval_labels = test_paths, test_labels
    
    # Create the dataset and dataloader for evaluation
    test_transform = get_val_transform()
    test_dataset = DigitSequenceDataset(eval_paths, eval_labels, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print(f"Running evaluation on {len(test_dataset)} test samples...")
    start_time = time.time()
    results = evaluate_model(model, test_loader, temp_scaler)
    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    
    # Generate report
    print("Generating evaluation report...")
    report_dir = generate_report(results, REPORTS_DIR, temp_scaler)
    
    # Print key metrics
    print("\n" + "="*60)
    print(f"Test Samples: {results['total_samples']}")
    print(f"Per-Digit Accuracy: {results['digit_accuracy']:.2f}%")
    print(f"Full Sequence Accuracy: {results['sequence_accuracy']:.2f}%")
    print("="*60)
    
    # Generate README table entry
    table_entry = generate_evaluation_table_for_readme(results, report_dir)
    print("\nEntry for README Ongoing Evaluation Results table:")
    print(table_entry)
    
    # Save table entry to a file for easy copying
    with open(os.path.join(report_dir, 'readme_table_entry.md'), 'w') as f:
        f.write(table_entry)
    
    print(f"\nEvaluation report saved to: {report_dir}")
    print(f"To view the full report, open: {os.path.join(report_dir, 'evaluation_report.md')}")


if __name__ == "__main__":
    main() 