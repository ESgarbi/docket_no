#!/usr/bin/env python3
"""
Comparison script for evaluating different model predictions.

This script takes an image file path as input, runs predictions using both the original
and calibrated models, and shows a side-by-side comparison of the results.

Usage: python compare_models.py --image path/to/image.jpg
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from termcolor import colored
from tabulate import tabulate

# Import from predict.py
from predict import load_model, load_temperature_scaler, preprocess_image, predict

# Constants
ORIGINAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 "outputs/final_model.pth")
CALIBRATED_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   "outputs/calibrated_model.pth")
TEMP_SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "outputs/temperature_scaler.pth")


def compare_predictions(image_path, original_model_path=None, calibrated_model_path=None, 
                      temp_scaler_path=None, output=None, visualize=True):
    """Compare predictions from original and calibrated models."""
    # Use default paths if not provided
    original_path = original_model_path if original_model_path else ORIGINAL_MODEL_PATH
    calibrated_path = calibrated_model_path if calibrated_model_path else CALIBRATED_MODEL_PATH
    scaler_path = temp_scaler_path if temp_scaler_path else TEMP_SCALER_PATH
    
    # Verify image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Load models
    print(f"Loading original model from: {original_path}")
    original_model = load_model(original_path)
    
    print(f"Loading calibrated model from: {calibrated_path}")
    calibrated_model = load_model(calibrated_path)
    
    # Load temperature scaler
    temp_scaler = load_temperature_scaler(original_model, scaler_path)
    
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    
    # Get predictions
    print("Running predictions...")
    original_results = predict(original_model, image_tensor, temp_scaler)
    calibrated_results = predict(calibrated_model, image_tensor, temp_scaler)
    
    # Display image
    if visualize:
        img = Image.open(image_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis('off')
        plt.show()
    
    # Compare sequences
    original_seq = original_results["sequence"]
    calibrated_seq = calibrated_results["sequence"]
    
    sequence_match = original_seq == calibrated_seq
    match_str = "MATCH" if sequence_match else "DIFFERENT"
    match_color = "green" if sequence_match else "red"
    
    print("\n" + "="*80)
    print(f"Sequence Comparison: {colored(match_str, match_color)}")
    print("="*80)
    print(f"Original Model: {original_seq}")
    print(f"Calibrated Model: {calibrated_seq}")
    print("="*80 + "\n")
    
    # Compare digit by digit
    comparison_data = []
    
    for i in range(len(original_results["digit_predictions"])):
        orig_digit = original_results["digit_predictions"][i]
        calib_digit = calibrated_results["digit_predictions"][i]
        
        pos = orig_digit["position"]
        orig_val = orig_digit["value"]
        calib_val = calib_digit["value"]
        orig_conf = orig_digit["confidence"]
        calib_conf = calib_digit["confidence"]
        orig_calib_conf = orig_digit["calibrated_confidence"]
        calib_calib_conf = calib_digit["calibrated_confidence"]
        
        # Determine if values match
        match = orig_val == calib_val
        
        # Confidence difference
        conf_diff = calib_conf - orig_conf
        calib_conf_diff = calib_calib_conf - orig_calib_conf
        
        comparison_data.append({
            "Position": pos,
            "Original Value": orig_val,
            "Calibrated Value": calib_val,
            "Match": "✓" if match else "✗",
            "Original Confidence": f"{orig_conf:.4f}",
            "Calibrated Confidence": f"{calib_conf:.4f}",
            "Confidence Difference": f"{conf_diff:+.4f}",
            "Original Calibrated Confidence": f"{orig_calib_conf:.4f}",
            "Calibrated Calibrated Confidence": f"{calib_calib_conf:.4f}",
            "Calibrated Confidence Difference": f"{calib_conf_diff:+.4f}"
        })
    
    # Display comparison table
    df = pd.DataFrame(comparison_data)
    print(tabulate(df, headers="keys", tablefmt="grid"))
    
    # Compare overall confidence metrics
    print("\n" + "="*80)
    print("Overall Confidence Metrics Comparison")
    print("="*80)
    orig_metrics = original_results["sequence_confidence"]
    calib_metrics = calibrated_results["sequence_confidence"]
    
    metrics_data = [
        ["Metric", "Original Model", "Calibrated Model", "Difference"],
        ["Min Confidence", f"{orig_metrics['min']:.4f}", f"{calib_metrics['min']:.4f}", 
         f"{calib_metrics['min'] - orig_metrics['min']:+.4f}"],
        ["Max Confidence", f"{orig_metrics['max']:.4f}", f"{calib_metrics['max']:.4f}", 
         f"{calib_metrics['max'] - orig_metrics['max']:+.4f}"],
        ["Avg Confidence", f"{orig_metrics['avg']:.4f}", f"{calib_metrics['avg']:.4f}", 
         f"{calib_metrics['avg'] - orig_metrics['avg']:+.4f}"],
        ["Min Calibrated Confidence", f"{orig_metrics['calibrated_min']:.4f}", 
         f"{calib_metrics['calibrated_min']:.4f}", 
         f"{calib_metrics['calibrated_min'] - orig_metrics['calibrated_min']:+.4f}"],
        ["Max Calibrated Confidence", f"{orig_metrics['calibrated_max']:.4f}", 
         f"{calib_metrics['calibrated_max']:.4f}", 
         f"{calib_metrics['calibrated_max'] - orig_metrics['calibrated_max']:+.4f}"],
        ["Avg Calibrated Confidence", f"{orig_metrics['calibrated_avg']:.4f}", 
         f"{calib_metrics['calibrated_avg']:.4f}", 
         f"{calib_metrics['calibrated_avg'] - orig_metrics['calibrated_avg']:+.4f}"]
    ]
    
    print(tabulate(metrics_data, headers="firstrow", tablefmt="grid"))
    
    # Save results if output path provided
    if output:
        comparison_results = {
            "image_path": image_path,
            "original_model": {
                "path": original_path,
                "prediction": original_results
            },
            "calibrated_model": {
                "path": calibrated_path,
                "prediction": calibrated_results
            },
            "comparison": {
                "sequence_match": sequence_match,
                "digit_comparison": comparison_data
            }
        }
        
        try:
            with open(output, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            print(f"\nComparison results saved to {output}")
        except Exception as e:
            print(f"Error saving comparison results: {e}")
    
    return (original_results, calibrated_results)

def main():
    """Main function to parse arguments and run comparison."""
    parser = argparse.ArgumentParser(description="Compare original and calibrated model predictions")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--original_model", help="Path to the original model (default: outputs/final_model.pth)")
    parser.add_argument("--calibrated_model", help="Path to the calibrated model (default: outputs/calibrated_model.pth)")
    parser.add_argument("--temp_scaler", help="Path to the temperature scaler (default: outputs/temperature_scaler.pth)")
    parser.add_argument("--output", help="Optional path to save comparison results as JSON")
    parser.add_argument("--no-visualize", action="store_true", help="Disable image visualization")
    args = parser.parse_args()
    
    compare_predictions(
        args.image, 
        args.original_model,
        args.calibrated_model,
        args.temp_scaler,
        args.output,
        not args.no_visualize
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 