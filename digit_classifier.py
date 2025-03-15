import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import time
import random
import json

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Simple CNN for 28x28 digit images
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x):
        return self.features(x)

# Custom dataset for loading digit images
class DigitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Collect all image paths and their corresponding labels
        for digit in range(10):
            digit_dir = os.path.join(root_dir, str(digit))
            if os.path.isdir(digit_dir):
                for img_name in os.listdir(digit_dir):
                    if img_name.endswith('.png'):
                        img_path = os.path.join(digit_dir, img_name)
                        self.samples.append((img_path, digit))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Training function
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Save predictions and targets for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    val_loss = running_loss / total
    val_acc = 100. * correct / total
    
    return val_loss, val_acc, all_preds, all_targets

# Visualize training progress
def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(all_preds, all_targets, save_path=None):
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Main training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, device, num_epochs=25, save_dir='models'):
    # Create directory for saving models if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, all_preds, all_targets = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_loss)  # For ReduceLROnPlateau
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print epoch results
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, os.path.join(save_dir, 'final_model.pth'))
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
    }
    
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    return train_losses, val_losses, train_accs, val_accs, all_preds, all_targets

# Feature extraction for future projects
def extract_features_batch(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Extracting features"):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_features = model.extract_features(inputs)
            
            # Average over spatial dimensions for each channel
            batch_features = nn.functional.adaptive_avg_pool2d(batch_features, (1, 1))
            batch_features = batch_features.view(batch_features.size(0), -1)
            
            features.append(batch_features.cpu().numpy())
            labels.append(targets.cpu().numpy())
    
    return np.vstack(features), np.concatenate(labels)

# Generate a detailed evaluation report
def generate_evaluation_report(model, test_loader, device, save_dir='evaluation'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Evaluate model
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Save predictions, targets, and probabilities
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100. * correct / total
    
    # Generate classification report
    report = classification_report(all_targets, all_preds, digits=4)
    
    # Save results
    with open(os.path.join(save_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(f'Accuracy: {accuracy:.2f}%\n\n')
        f.write('Classification Report:\n')
        f.write(report)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(all_preds, all_targets, save_path=os.path.join(save_dir, 'confusion_matrix.png'))
    
    # Calculate per-class accuracy
    per_class_acc = {}
    for i in range(10):
        class_indices = [idx for idx, label in enumerate(all_targets) if label == i]
        if class_indices:
            class_preds = [all_preds[idx] for idx in class_indices]
            class_targets = [all_targets[idx] for idx in class_indices]
            class_acc = sum(p == t for p, t in zip(class_preds, class_targets)) / len(class_indices)
            per_class_acc[i] = class_acc * 100
    
    # Plot per-class accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(per_class_acc.keys(), per_class_acc.values())
    plt.xlabel('Digit Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.xticks(range(10))
    plt.ylim(0, 100)
    plt.savefig(os.path.join(save_dir, 'per_class_accuracy.png'))
    
    # Save per-class accuracy
    with open(os.path.join(save_dir, 'per_class_accuracy.json'), 'w') as f:
        json.dump(per_class_acc, f)
    
    print(f'Evaluation completed. Accuracy: {accuracy:.2f}%')
    print(f'Detailed report saved to {save_dir}')
    
    return accuracy, report, per_class_acc

# Main function to run the training and evaluation
def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    print(f'Device: {device}')
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Dataset and DataLoader
    dataset_path = 'datasets/pod_digits'
    
    # Create full dataset with training transforms
    full_dataset = DigitDataset(root_dir=dataset_path, transform=train_transform)
    
    # Split dataset into train, validation, and test sets (70%, 15%, 15%)
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size])
    
    # Override transforms for validation and test datasets
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    print(f'Dataset sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}')
    
    # Initialize model
    model = DigitClassifier().to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # Train the model
    print("Starting training...")
    train_losses, val_losses, train_accs, val_accs, all_preds, all_targets = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        device, num_epochs=35, save_dir='models')
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                         save_path='models/training_curves.png')
    
    # Plot confusion matrix
    plot_confusion_matrix(all_preds, all_targets, save_path='models/confusion_matrix.png')
    
    # Load best model for evaluation
    checkpoint = torch.load('models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate evaluation report
    print("\nGenerating evaluation report...")
    generate_evaluation_report(model, test_loader, device, save_dir='evaluation')
    
    # Extract features example (useful for future projects)
    print("\nExtracting features as an example for future projects...")
    features, labels = extract_features_batch(model, test_loader, device)
    print(f"Extracted features shape: {features.shape}")
    
    # Save feature extraction example
    np.save('models/example_features.npy', features)
    np.save('models/example_labels.npy', labels)
    
    print("\nTraining and evaluation completed successfully!")

if __name__ == "__main__":
    main() 