#!/usr/bin/env python3
"""
Batch Comparison Script for Model Evaluation.

This script compares the predictions of the original and calibrated models across
multiple test images and generates overall statistics.

Usage: python batch_compare.py --test_dir path/to/test/dir --num_samples 50
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from pathlib import Path
from datetime import datetime

# Import the comparison function
from compare_models import compare_predictions

def run_batch_comparison(test_dir, num_samples=50, random_seed=42, output_dir=None):
    """Run batch comparison on multiple test images."""
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Get all image files from test directory
    image_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    print(f"Found {len(image_files)} image files in {test_dir}")
    
    # Sample images if requested
    if num_samples and num_samples < len(image_files):
        image_files = random.sample(image_files, num_samples)
        print(f"Randomly selected {num_samples} images for comparison")
    
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = os.path.join(output_dir, f"batch_comparison_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = None
    
    # Initialize results
    total_images = len(image_files)
    sequence_matches = 0
    digit_matches = 0
    total_digits = 0
    confidence_diffs = []
    calibrated_confidence_diffs = []
    wrong_predictions_original = 0
    wrong_predictions_calibrated = 0
    position_errors = {i+1: {'original': 0, 'calibrated': 0} for i in range(13)}
    
    # Process each image
    print(f"Comparing predictions on {total_images} images...")
    image_results = []
    
    for i, image_file in enumerate(tqdm(image_files, desc="Processing images")):
        image_path = os.path.join(test_dir, image_file)
        
        # Set output path if specified
        if results_dir:
            image_output = os.path.join(results_dir, f"{image_file[:-4]}_comparison.json")
        else:
            image_output = None
        
        # Run comparison
        original_results, calibrated_results = compare_predictions(
            image_path, 
            output=image_output,
            visualize=False
        )
        
        # Extract sequences
        original_seq = original_results["sequence"]
        calibrated_seq = calibrated_results["sequence"]
        
        # Check if sequences match
        sequence_match = original_seq == calibrated_seq
        if sequence_match:
            sequence_matches += 1
        
        # Compare digits
        for j in range(len(original_results["digit_predictions"])):
            total_digits += 1
            orig_digit = original_results["digit_predictions"][j]
            calib_digit = calibrated_results["digit_predictions"][j]
            
            # Count digit matches
            if orig_digit["value"] == calib_digit["value"]:
                digit_matches += 1
            else:
                # Track position-specific errors
                position = orig_digit["position"]
                position_errors[position]['original'] += 1
                position_errors[position]['calibrated'] += 1
            
            # Track confidence differences
            conf_diff = calib_digit["confidence"] - orig_digit["confidence"]
            cal_conf_diff = calib_digit["calibrated_confidence"] - orig_digit["calibrated_confidence"]
            confidence_diffs.append(conf_diff)
            calibrated_confidence_diffs.append(cal_conf_diff)
        
        # Store results
        image_results.append({
            "image": image_file,
            "original_sequence": original_seq,
            "calibrated_sequence": calibrated_seq,
            "sequence_match": sequence_match,
            "original_confidence": original_results["sequence_confidence"],
            "calibrated_confidence": calibrated_results["sequence_confidence"]
        })
    
    # Calculate overall statistics
    sequence_match_rate = sequence_matches / total_images
    digit_match_rate = digit_matches / total_digits
    avg_confidence_diff = np.mean(confidence_diffs)
    avg_calibrated_confidence_diff = np.mean(calibrated_confidence_diffs)
    
    # Position-specific error analysis
    position_error_rates = []
    for pos, counts in position_errors.items():
        position_error_rates.append({
            "Position": pos,
            "Original Errors": counts["original"],
            "Calibrated Errors": counts["calibrated"],
            "Difference": counts["original"] - counts["calibrated"]
        })
    
    # Create summary report
    summary = {
        "overall": {
            "total_images": total_images,
            "sequence_matches": sequence_matches,
            "sequence_match_rate": sequence_match_rate,
            "digit_matches": digit_matches,
            "total_digits": total_digits,
            "digit_match_rate": digit_match_rate,
            "avg_confidence_diff": avg_confidence_diff,
            "avg_calibrated_confidence_diff": avg_calibrated_confidence_diff
        },
        "position_errors": position_error_rates,
        "image_results": image_results
    }
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH COMPARISON SUMMARY")
    print("="*80)
    print(f"Total Images: {total_images}")
    print(f"Sequence Match Rate: {sequence_match_rate:.2%} ({sequence_matches}/{total_images})")
    print(f"Digit Match Rate: {digit_match_rate:.2%} ({digit_matches}/{total_digits})")
    print(f"Average Confidence Difference: {avg_confidence_diff:+.4f}")
    print(f"Average Calibrated Confidence Difference: {avg_calibrated_confidence_diff:+.4f}")
    
    # Print position-specific errors
    print("\n" + "="*80)
    print("POSITION-SPECIFIC ERROR ANALYSIS")
    print("="*80)
    print(tabulate(position_error_rates, headers="keys", tablefmt="grid"))
    
    # Save summary if output directory specified
    if results_dir:
        summary_path = os.path.join(results_dir, "batch_summary.json")
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved to {summary_path}")
        except Exception as e:
            print(f"Error saving summary: {e}")
    
    return summary

def main():
    """Main function to parse arguments and run batch comparison."""
    parser = argparse.ArgumentParser(description="Run batch comparison of model predictions")
    parser.add_argument("--test_dir", default="/Users/erick/git/prod/docket_no/sequence/001/test",
                      help="Directory containing test images")
    parser.add_argument("--num_samples", type=int, default=50,
                      help="Number of random images to sample (0 for all)")
    parser.add_argument("--random_seed", type=int, default=42,
                      help="Random seed for sampling")
    parser.add_argument("--output_dir", default="comparison_results",
                      help="Directory to save results")
    args = parser.parse_args()
    
    run_batch_comparison(
        args.test_dir,
        args.num_samples,
        args.random_seed,
        args.output_dir
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 