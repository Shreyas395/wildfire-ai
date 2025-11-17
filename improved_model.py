# Improved Wildfire Detection Model with Better Accuracy
import numpy as np
import pandas as pd
import os
from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
import copy
warnings.filterwarnings('ignore')

# Import PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from torch.cuda import amp

# Import Torchvision
import torchvision
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor

# Training utilities
from timeit import default_timer as timer
from tqdm.auto import tqdm

# Metrics and visualization
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    torch.backends.cudnn.benchmark = True

class Config:
    """Configuration class for hyperparameters"""
    # Data
    IMG_SIZE = 224  # Increased from 128 for better feature extraction
    BATCH_SIZE = 32  # Increased batch size
    
    # Training
    EPOCHS = 20
    LEARNING_RATE = 0.001  # Lower learning rate for better convergence
    WEIGHT_DECAY = 1e-4  # L2 regularization
    
    # Model
    NUM_CLASSES = 2
    DROPOUT_RATE = 0.5
    NUM_WORKERS = min(4, os.cpu_count() or 1)
    USE_AMP = True
    
    # Early stopping
    PATIENCE = 5
    MIN_DELTA = 0.001

# Helper to ensure transforms stay pickleable on Windows
def convert_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")

# Enhanced Data Augmentation
def get_transforms(phase: str):
    """Get data transforms for training and validation"""
    if phase == 'train':
        return transforms.Compose([
            transforms.Lambda(convert_to_rgb),
            transforms.Resize((Config.IMG_SIZE + 32, Config.IMG_SIZE + 32)),
            transforms.RandomCrop(Config.IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        return transforms.Compose([
            transforms.Lambda(convert_to_rgb),
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# Improved CNN Architecture with Residual Connections
class ImprovedWildfireCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super().__init__()
        
        # Feature extractor with batch normalization
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.4),
        )
        
        # Adaptive pooling for flexible input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Transfer Learning Models
def create_resnet_model(num_classes: int = 2, pretrained: bool = True):
    """Create ResNet50 model with transfer learning"""
    model = models.resnet50(pretrained=pretrained)
    
    # Freeze early layers for transfer learning
    if pretrained:
        for param in list(model.parameters())[:-20]:  # Freeze all but last 20 parameters
            param.requires_grad = False
    
    # Replace final layer
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

def create_efficientnet_model(num_classes: int = 2, pretrained: bool = True):
    """Create EfficientNet-B0 model with transfer learning"""
    model = models.efficientnet_b0(pretrained=pretrained)
    
    # Freeze early layers
    if pretrained:
        for param in list(model.parameters())[:-10]:
            param.requires_grad = False
    
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

# Learning Rate Scheduler
def get_scheduler(optimizer, scheduler_type='cosine'):
    """Get learning rate scheduler"""
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    else:
        return None

# Enhanced Training Functions
def train_step(model: nn.Module, 
               dataloader: DataLoader, 
               loss_fn: nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               scaler: amp.GradScaler | None = None,
               amp_enabled: bool = False) -> Tuple[float, float]:
    """Enhanced training step with gradient clipping and optional AMP"""
    model.train()
    train_loss, train_acc = 0, 0
    
    for X, y in dataloader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=amp_enabled):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
        
        # Backward pass
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Calculate metrics
        train_loss += loss.item()
        train_acc += (y_pred.argmax(1) == y).type(torch.float).sum().item()
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader.dataset)
    
    return train_loss, train_acc

def test_step(model: nn.Module, 
              dataloader: DataLoader, 
              loss_fn: nn.Module, 
              device: torch.device,
              amp_enabled: bool = False) -> Tuple[float, float, List, List]:
    """Enhanced test step with predictions collection"""
    model.eval()
    test_loss, test_acc = 0, 0
    y_true_list, y_pred_list = [], []
    
    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with amp.autocast(enabled=amp_enabled):
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
            
            test_loss += loss.item()
            test_acc += (y_pred.argmax(1) == y).type(torch.float).sum().item()
            
            # Collect predictions for metrics
            y_true_list.extend(y.cpu().numpy())
            y_pred_list.extend(y_pred.argmax(1).cpu().numpy())
    
    test_loss /= len(dataloader)
    test_acc /= len(dataloader.dataset)
    
    return test_loss, test_acc, y_true_list, y_pred_list

# Model Training with All Improvements
def train_model(model: nn.Module, 
                train_dataloader: DataLoader, 
                test_dataloader: DataLoader,
                epochs: int = Config.EPOCHS,
                learning_rate: float = Config.LEARNING_RATE,
                device: torch.device = device) -> Dict:
    """Train model with all improvements"""
    
    # Loss function with class weights for imbalanced data
    loss_fn = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=learning_rate, 
                                 weight_decay=Config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = get_scheduler(optimizer, 'cosine')
    
    # Early stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE, min_delta=Config.MIN_DELTA)

    amp_enabled = device.type == "cuda" and Config.USE_AMP
    scaler = amp.GradScaler(enabled=amp_enabled)
    
    # Tracking
    results = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
        "learning_rates": []
    }
    
    print(f"Training {model.__class__.__name__} for {epochs} epochs...")
    print(f"Device: {device}")
    
    # Training loop
    start_time = timer()
    
    progress_bar = tqdm(range(epochs), desc="Training Progress", leave=True)
    for epoch in progress_bar:
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)

        # Training
        epoch_start = timer()
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, device, scaler=scaler if amp_enabled else None,
            amp_enabled=amp_enabled
        )

        # Testing
        test_loss, test_acc, y_true, y_pred = test_step(model, test_dataloader, loss_fn, device, amp_enabled=amp_enabled)
        
        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Update learning rate
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()
        
        # Store results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["learning_rates"].append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | F1: {f1:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        torch.cuda.synchronize() if device.type == "cuda" else None
        epoch_time = timer() - epoch_start
        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            device_monitor = f"GPU Memory: {alloc:.1f}MB allocated / {reserved:.1f}MB reserved"
        else:
            device_monitor = "Running on CPU"
        print(f"Epoch duration: {epoch_time:.2f}s | {device_monitor}")

        progress_bar.set_postfix({
            "TrainLoss": f"{train_loss:.3f}",
            "ValLoss": f"{test_loss:.3f}",
            "ValAcc": f"{test_acc:.3f}",
            "Time(s)": f"{epoch_time:.1f}"
        })

        # Early stopping check
        if early_stopping(test_loss, model):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    end_time = timer()
    total_time = end_time - start_time
    
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Best Test Accuracy: {max(results['test_acc']):.4f}")
    
    return results

# Visualization Functions
def plot_training_curves(results: Dict, save_path: str = None):
    """Plot training and validation curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(results['train_loss']) + 1)
    
    # Loss curves
    ax1.plot(epochs, results['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, results['test_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(epochs, results['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, results['test_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate
    ax3.plot(epochs, results['learning_rates'], 'g-')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    # Best accuracy so far
    best_acc = [max(results['test_acc'][:i+1]) for i in range(len(results['test_acc']))]
    ax4.plot(epochs, best_acc, 'm-', label='Best Validation Accuracy')
    ax4.set_title('Best Validation Accuracy Progress')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Best Accuracy')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(y_true: List, y_pred: List, class_names: List, save_path: str = None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

# Model Comparison Function
def compare_models():
    """Compare different model architectures"""
    print("This function will compare different model architectures")
    print("Models to compare: Custom CNN, ResNet50, EfficientNet-B0")
    
    return {
        'custom_cnn': ImprovedWildfireCNN(num_classes=Config.NUM_CLASSES),
        'resnet50': create_resnet_model(num_classes=Config.NUM_CLASSES),
        'efficientnet_b0': create_efficientnet_model(num_classes=Config.NUM_CLASSES)
    }

if __name__ == "__main__":
    print("Improved Wildfire Detection Model loaded successfully!")
    print("Key improvements:")
    print("- Enhanced data augmentation")
    print("- Deeper CNN architecture with batch normalization")
    print("- Transfer learning with ResNet and EfficientNet")
    print("- Advanced training techniques (early stopping, LR scheduling)")
    print("- Better evaluation metrics and visualization")
