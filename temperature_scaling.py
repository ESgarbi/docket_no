#!/usr/bin/env python3
"""
Temperature Scaling for calibrating confidence scores of the digit sequence model.
This module provides functionality to learn optimal temperature values and apply them to model outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
import numpy as np
from tqdm import tqdm

class DigitTemperatureScaler:
    """
    Temperature scaling for multi-digit sequence recognition.
    Learns a separate temperature parameter for each digit position.
    """
    def __init__(self, model, num_digits=13, device='cuda'):
        self.model = model
        self.num_digits = num_digits
        self.device = device
        # Initialize separate temperature parameter for each digit position
        self.temperatures = nn.Parameter(torch.ones(num_digits, device=device))
        self.optimized = False
    
    def _get_digit_temperatures(self):
        return {f'digit_{i+1}': self.temperatures[i].item() for i in range(self.num_digits)}
    
    def optimize_temperatures(self, valid_loader, max_iter=100):
        """Learn the optimal temperature values using the validation set."""
        self.model.eval()
        digit_positions = [f'digit_{i+1}' for i in range(self.num_digits)]
        optimizer = LBFGS([self.temperatures], lr=0.01, max_iter=max_iter)
        
        # Store all validation logits and labels for optimization
        logits_list = {pos: [] for pos in digit_positions}
        labels_list = {pos: [] for pos in digit_positions}
        
        # First, collect all the logits and labels from the validation set
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                
                for i, pos in enumerate(digit_positions):
                    if pos in outputs:
                        logits_list[pos].append(outputs[pos].detach())
                        labels_list[pos].append(targets[:, i])
        
        # Concatenate all the logits and labels
        all_logits = {}
        all_labels = {}
        for pos in digit_positions:
            if logits_list[pos]:
                all_logits[pos] = torch.cat(logits_list[pos])
                all_labels[pos] = torch.cat(labels_list[pos])
        
        def eval_loss():
            loss = 0.0
            for i, pos in enumerate(digit_positions):
                if pos in all_logits:
                    # Re-enable gradient computation for the temperature parameter
                    scaled_logits = all_logits[pos] / self.temperatures[i]
                    log_probs = F.log_softmax(scaled_logits, dim=1)
                    pos_loss = F.nll_loss(log_probs, all_labels[pos])
                    loss += pos_loss
            
            return loss / len(digit_positions)
        
        def closure():
            optimizer.zero_grad()
            loss = eval_loss()
            loss.backward()
            return loss
        
        print("Optimizing temperature parameters...")
        optimizer.step(closure)
        self.optimized = True
        
        # Print the optimized temperatures
        print("Optimized temperature values:")
        for i, t in enumerate(self.temperatures):
            print(f"  Digit {i+1}: {t.item():.4f}")
        
        return self._get_digit_temperatures()
    
    def calibrate_confidence(self, logits, position_idx):
        """
        Apply temperature scaling to logits for a specific digit position.
        Args:
            logits: Raw model outputs (before softmax) for a specific digit position
            position_idx: Index of the digit position (0-based)
        
        Returns:
            Calibrated probabilities after temperature scaling and softmax
        """
        if not self.optimized:
            print("Warning: Using default temperature values. Run optimize_temperatures first.")
        
        with torch.no_grad():
            # Apply temperature scaling
            temperature = self.temperatures[position_idx]
            scaled_logits = logits / temperature
            probabilities = F.softmax(scaled_logits, dim=1)
        
        return probabilities
    
    def calibrate_outputs(self, outputs):
        """
        Apply temperature scaling to all outputs of the model.
        
        Args:
            outputs: Dictionary of model outputs with keys like 'digit_1', 'digit_2', etc.
            
        Returns:
            Dictionary with calibrated probabilities
        """
        calibrated = {}
        for i in range(self.num_digits):
            pos = f'digit_{i+1}'
            if pos in outputs:
                logits = outputs[pos]
                calibrated[pos] = self.calibrate_confidence(logits, i)
        
        return calibrated
    
    def save(self, path):
        """Save the temperature parameters to a file."""
        torch.save({
            'temperatures': self.temperatures.detach().cpu(),
            'optimized': self.optimized
        }, path)
    
    def load(self, path):
        """Load temperature parameters from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.temperatures = nn.Parameter(checkpoint['temperatures'].to(self.device))
        self.optimized = checkpoint['optimized']
        return self._get_digit_temperatures()


def calibrate_model(model, valid_loader, num_digits=13, device='cuda'):
    """
    Convenience function to create and optimize a temperature scaler.
    
    Args:
        model: The trained model to calibrate
        valid_loader: DataLoader for validation set
        num_digits: Number of digits in the sequence
        device: Device to use for computation
        
    Returns:
        Calibrated model and temperature values
    """
    scaler = DigitTemperatureScaler(model, num_digits=num_digits, device=device)
    temperatures = scaler.optimize_temperatures(valid_loader)
    
    # Save the temperature scaler
    scaler.save('outputs/temperature_scaler.pth')
    
    return scaler, temperatures 