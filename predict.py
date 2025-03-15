#!/usr/bin/env python3
"""
Prediction script for Docket Number Sequence Recognition.

This script takes an image file path as input, runs the digit sequence recognition model,
and returns a JSON with the predicted sequence and confidence scores.

Usage: python predict.py --image path/to/image.jpg
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

# Import the model and temperature scaling classes
from google_colab_training_resnet import ResNetDigitSequence
from temperature_scaling import DigitTemperatureScaler

# Constants
NUM_DIGITS = 13
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                     'mps' if torch.backends.mps.is_available() else 
                     'cpu')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs/final_model.pth")
TEMP_SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs/temperature_scaler.pth")

def load_model(model_path=None):
    """Load the trained model."""
    try:
        # Use specified model path or default
        path = model_path if model_path else MODEL_PATH
        
        model = ResNetDigitSequence()
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def load_temperature_scaler(model, temp_scaler_path=None):
    """Load the temperature scaler for confidence calibration."""
    # Use specified temperature scaler path or default
    path = temp_scaler_path if temp_scaler_path else TEMP_SCALER_PATH
    
    if not os.path.exists(path):
        print(f"Warning: Temperature scaler not found at {path}. "
              "Using uncalibrated confidence scores.")
        return None
    
    try:
        temp_scaler = DigitTemperatureScaler(model, num_digits=NUM_DIGITS, device=DEVICE)
        temp_scaler.load(path)
        return temp_scaler
    except Exception as e:
        print(f"Warning: Could not load temperature scaler: {e}. "
              "Using uncalibrated confidence scores.")
        return None

def preprocess_image(image_path):
    """Preprocess the input image for the model."""
    try:
        # Open and verify the image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply the same transformations used during validation/testing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Transform and add batch dimension
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        return img_tensor
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        sys.exit(1)

def predict(model, image_tensor, temp_scaler=None):
    """Run inference on the image and return predictions with confidence scores."""
    digit_positions = [f'digit_{i+1}' for i in range(NUM_DIGITS)]
    
    with torch.no_grad():
        # Get model outputs
        outputs = model(image_tensor)
        
        # Apply temperature scaling if available
        if temp_scaler:
            calibrated_outputs = temp_scaler.calibrate_outputs(outputs)
        
        # Initialize results dictionary
        results = {
            "sequence": "",
            "digit_predictions": [],
            "calibrated": temp_scaler is not None
        }
        
        # Process each digit position
        for i, pos in enumerate(digit_positions):
            if pos in outputs:
                # Get original confidence
                logits = outputs[pos][0]  # First (only) item in batch
                probs = F.softmax(logits, dim=0).cpu().numpy()
                pred = int(np.argmax(probs))
                confidence = float(probs[pred])
                
                # Get calibrated confidence if available
                calibrated_confidence = None
                if temp_scaler and pos in calibrated_outputs:
                    cal_probs = calibrated_outputs[pos][0].cpu().numpy()
                    calibrated_confidence = float(cal_probs[pred])
                
                # Add to results
                results["sequence"] += str(pred)
                results["digit_predictions"].append({
                    "position": i + 1,
                    "value": pred,
                    "confidence": confidence,
                    "calibrated_confidence": calibrated_confidence
                })
    
    # Calculate overall sequence confidence metrics
    confidence_values = [d["confidence"] for d in results["digit_predictions"]]
    
    if temp_scaler:
        calibrated_values = [d["calibrated_confidence"] for d in results["digit_predictions"] 
                          if d["calibrated_confidence"] is not None]
        results["sequence_confidence"] = {
            "min": min(confidence_values),
            "max": max(confidence_values),
            "avg": sum(confidence_values) / len(confidence_values),
            "calibrated_min": min(calibrated_values) if calibrated_values else None,
            "calibrated_max": max(calibrated_values) if calibrated_values else None,
            "calibrated_avg": sum(calibrated_values) / len(calibrated_values) if calibrated_values else None
        }
    else:
        results["sequence_confidence"] = {
            "min": min(confidence_values),
            "max": max(confidence_values),
            "avg": sum(confidence_values) / len(confidence_values)
        }
    
    return results

def main():
    """Main function to parse arguments and run prediction."""
    parser = argparse.ArgumentParser(description="Run digit sequence prediction on a single image")
    parser.add_argument("--image", default="sequence/current_samples_off_test/account_no/25075004.png", required=False, help="Path to the image file")
    parser.add_argument("--output", help="Optional path to save JSON output")
    parser.add_argument("--no-calibration", action="store_true", 
                      help="Disable confidence calibration with temperature scaling")
    parser.add_argument("--model_path", default="/Users/erick/git/prod/docket_no/outputs/calibrated_model.pth", help="Path to the model file (default: outputs/final_model.pth)")
    parser.add_argument("--temp_scaler_path", help="Path to the temperature scaler file (default: outputs/temperature_scaler.pth)")
    args = parser.parse_args()
    
    # Verify image path exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Load model
    model = load_model(args.model_path)
    
    # Load temperature scaler if not disabled
    temp_scaler = None
    if not args.no_calibration:
        temp_scaler = load_temperature_scaler(model, args.temp_scaler_path)
    
    # Preprocess image
    image_tensor = preprocess_image(args.image)
    
    # Run prediction
    results = predict(model, image_tensor, temp_scaler)
    
    # Format JSON output
    json_output = json.dumps(results, indent=2)
    
    # Save to file if specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(json_output)
            print(f"Prediction results saved to {args.output}")
        except Exception as e:
            print(f"Error saving results to file: {e}")
    
    # Print to console
    print(json_output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 