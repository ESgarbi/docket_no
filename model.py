import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self):
        super(ResNetDigitSequence, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Create ResNet layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_dim = 256
        
        # Create 13 separate digit classifiers
        self.digit_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.flatten_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 10)
            ) for _ in range(13)
        ])
        
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
        
    def forward(self, x):
        # Extract features
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply each digit classifier
        digit_outputs = {}
        for i, classifier in enumerate(self.digit_classifiers):
            digit_outputs[f'digit_{i+1}'] = classifier(x)
            
        return digit_outputs
    


    ### --->> Create a custom loss that targets all 13 digits separately

class DigitSequenceLoss(nn.Module):
    """
    Custom loss function for 13-digit sequence recognition.
    Calculates separate losses for each digit position and allows for weighting.
    """
    def __init__(self, num_digits=13, weights=None, use_sequence_penalty=False, 
                 sequence_lambda=0.1, focal_gamma=0.0):
        """
        Initialize the DigitSequenceLoss.
        
        Args:
            num_digits (int): Number of digits in the sequence (default: 13)
            weights (list): Weights for each digit position loss (default: equal weights)
            use_sequence_penalty (bool): Whether to add a sequence consistency penalty
            sequence_lambda (float): Weight for the sequence consistency penalty
            focal_gamma (float): Gamma parameter for focal loss (0 = standard CE loss)
        """
        super(DigitSequenceLoss, self).__init__()
        self.num_digits = num_digits
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.use_sequence_penalty = use_sequence_penalty
        self.sequence_lambda = sequence_lambda
        self.focal_gamma = focal_gamma
        
        # Default: equal weights for all digit positions
        if weights is None:
            self.weights = [1.0] * num_digits
        else:
            assert len(weights) == num_digits, "Weights must match number of digits"
            self.weights = weights
            
        # Normalize weights to sum to number of digits
        weight_sum = sum(self.weights)
        self.weights = [w * self.num_digits / weight_sum for w in self.weights]
        
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
        
        for i, pos in enumerate(digit_positions):
            if pos in outputs:
                # Get predictions and targets for this digit position
                digit_preds = outputs[pos]
                digit_targets = targets[:, i]
                
                # Calculate standard cross entropy loss
                ce_loss = self.ce_loss(digit_preds, digit_targets)
                
                # Apply focal loss if gamma > 0
                if self.focal_gamma > 0:
                    probs = torch.gather(
                        F.softmax(digit_preds, dim=1), 
                        dim=1, 
                        index=digit_targets.unsqueeze(1)
                    ).squeeze(1)
                    focal_weight = (1 - probs) ** self.focal_gamma
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
        
        # Apply sequence consistency penalty if enabled
        if self.use_sequence_penalty and len(all_probs) > 1:
            sequence_penalty = 0.0
            
            # Calculate pairwise probability distribution differences between adjacent positions
            for i in range(len(all_probs) - 1):
                # KL divergence to penalize dramatically different distributions
                # between adjacent digit positions
                sequence_penalty += F.kl_div(
                    all_probs[i].log(), all_probs[i+1], 
                    reduction='batchmean'
                )
            
            # Add sequence penalty to total loss
            sequence_penalty = sequence_penalty / (len(all_probs) - 1)
            total_loss = total_loss + self.sequence_lambda * sequence_penalty
            
            # Store sequence penalty for reporting
            digit_losses['sequence_penalty'] = sequence_penalty.item()
        
        # Return both the total loss and individual losses
        return total_loss / self.num_digits, digit_losses

# Example usage:
# criterion = DigitSequenceLoss()
# loss, digit_losses = criterion(model_outputs, target_labels)

# Example with weighted loss (emphasizing later digits):
"""
# Basic usage with default settings
criterion = DigitSequenceLoss()

# Progressive weighting (increasing weights for later digits)
weights = [0.5 + i * 0.1 for i in range(13)]  # [0.5, 0.6, 0.7, ..., 1.7]
criterion_weighted = DigitSequenceLoss(weights=weights)

# With sequence consistency penalty
criterion_with_penalty = DigitSequenceLoss(
    use_sequence_penalty=True,
    sequence_lambda=0.2  # Adjust the weight of the sequence penalty
)

# With focal loss for handling imbalanced classes
criterion_focal = DigitSequenceLoss(
    focal_gamma=2.0  # Higher values focus more on hard examples
)

# Combining all advanced features
criterion_advanced = DigitSequenceLoss(
    weights=weights,
    use_sequence_penalty=True,
    sequence_lambda=0.2,
    focal_gamma=2.0
)

# Using the criterion in training
model = ResNetDigitSequence()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Choose which loss function to use
        loss, digit_losses = criterion_advanced(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        # You can monitor individual digit losses
        for pos, digit_loss in digit_losses.items():
            print(f"{pos}: {digit_loss:.4f}", end=" | ")
        print()  # Newline after printing all losses
"""