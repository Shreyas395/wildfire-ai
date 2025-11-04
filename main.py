# Early Wildfire Detection Model - Enhanced Version
import numpy as np
import pandas as pd
import os
from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader

# Import Torchvision
import torchvision
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor

# Import improved model components
from improved_model import (
    Config, get_transforms, ImprovedWildfireCNN, 
    create_resnet_model, create_efficientnet_model,
    train_model, plot_training_curves, plot_confusion_matrix
)

# Check version
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

# Setup Device Agnostic Code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Access and walkthrough Data
data_path = Path("data/")
image_path = data_path / "the_wildfire_dataset_2n_version/"

# If data path doesn't exist, try alternative paths
if not image_path.exists():
    data_path = Path("./data/")
    image_path = data_path / "the_wildfire_dataset_2n_version/"

if not image_path.exists():
    print(f"Warning: Data path {image_path} does not exist.")
    print("Please ensure the wildfire dataset is in the correct location.")
    # For now, create dummy directories to avoid crashes
    image_path.mkdir(parents=True, exist_ok=True)
    (image_path / "train").mkdir(exist_ok=True)
    (image_path / "test").mkdir(exist_ok=True)
else:
    def walk_through_dir(dir_path):
        for dirpath, dirnames, filenames in os.walk(dir_path):
            print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

    walk_through_dir(image_path)

train_dir = image_path / "train"
test_dir = image_path / "test"
print(f"Train directory: {train_dir}")
print(f"Test directory: {test_dir}")

import random
from PIL import Image

# Set Random Seed
random.seed(42)

# 1. Obtain Image Paths (only if data exists)
if image_path.exists():
    image_path_list = list(image_path.glob("*/*/*.jpg"))
    if len(image_path_list) > 0:
        # 2. Pick Random Image Path
        random_image_path = random.choice(image_path_list)

        # 3. Get Image Class (e.g. Fire or No Fire)
        image_class = random_image_path.parent.stem

        # 4. Open & View Image
        img = Image.open(random_image_path)

        # 5. Print MetaData
        print(f"Random Image Path: {random_image_path}")
        print(f"Image Class: {image_class}")
        print(f"Image Height: {img.height}")
        print(f"Image Width: {img.width}")
    else:
        print("No JPG images found in the dataset.")
        image_path_list = []
else:
    image_path_list = []

# Convert image to array and display (only if we have an image)
if 'img' in locals():
    img_as_array = np.asarray(img)
    print(f"Image array shape: {img_as_array.shape}")
    print(f"Image array dtype: {img_as_array.dtype}")
else:
    print("No image loaded to convert to array.")

# Enhanced Data transformation pipeline
print("\n=== Data Transformation Setup ===")
train_transform = get_transforms('train')
test_transform = get_transforms('test')

print(f"Using image size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
print("Enhanced augmentations applied for training data")

# Test transform on sample image (if available)
if 'img' in locals():
    transformed_img = train_transform(img)
    print(f"Transformed image shape: {transformed_img.shape}")
    print(f"Transformed image dtype: {transformed_img.dtype}")
else:
    print("No image available to test transformation.")

def plot_transformed_images(image_paths: list, transform, n=3, seed=None):
    """
    Selects random images from image paths and plots the original vs the transformed image
    """
    if len(image_paths) == 0:
        print("No images available to plot.")
        return

    if seed:
        random.seed(seed)

    # Ensure we don't sample more images than available
    n = min(n, len(image_paths))
    random_image_paths = random.sample(image_paths, k=n)

    for image_path in random_image_paths:
        with Image.open(image_path) as f:

            # Plot Original Image
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis("off")

            # Plot Transformed Image
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
            plt.show()

# Plot transformed images (only if we have images)
if len(image_path_list) > 0:
    plot_transformed_images(image_paths=image_path_list,
                          transform=train_transform,  # Use enhanced transforms
                          n=3,
                          seed=42)
else:
    print("No images available for plotting.")

# Create datasets (only if data directories exist)
if train_dir.exists() and test_dir.exists() and len(list(train_dir.glob("*/*.jpg"))) > 0:
    try:
        train_data = datasets.ImageFolder(root=train_dir,
                                        transform=train_transform,  # Enhanced training transforms
                                        target_transform=None)
        test_data = datasets.ImageFolder(root=test_dir,
                                       transform=test_transform,   # Enhanced test transforms
                                       target_transform=None)

        # Get All Class Names as a List
        class_names = train_data.classes
        class_dict = train_data.class_to_idx

        print(f"Class names: {class_names}")
        print(f"Class to index mapping: {class_dict}")
        print(f"Training samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

    except Exception as e:
        print(f"Error creating datasets: {e}")
        # Create dummy data for testing
        class_names = ['fire', 'no_fire']
        class_dict = {'fire': 0, 'no_fire': 1}
        train_data = None
        test_data = None
else:
    print("Training or test data directories not found or empty.")
    class_names = ['fire', 'no_fire']  # Default classes
    class_dict = {'fire': 0, 'no_fire': 1}
    train_data = None
    test_data = None

# Create DataLoader (only if we have valid data)
if train_data is not None and test_data is not None:
    # Use enhanced batch size from config
    BATCH_SIZE = Config.BATCH_SIZE
    print(f"\n=== DataLoader Setup ===")
    print(f"Using batch size: {BATCH_SIZE}")

    # Create Iterable Datasets (DataLoader)
    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                               batch_size=BATCH_SIZE,
                               shuffle=False)

    print(f"DataLoaders created successfully.")
    print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
    print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

    # Test a batch
    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features_batch.shape}")
    print(f"Labels batch shape: {train_labels_batch.shape}")
else:
    print("No valid data available for DataLoader creation.")
    train_dataloader = None
    test_dataloader = None
    BATCH_SIZE = 16

# Model Selection and Initialization
print("\n=== Model Architecture Selection ===")
print("Available models:")
print("1. Improved Custom CNN (Enhanced)")
print("2. ResNet50 (Transfer Learning)")
print("3. EfficientNet-B0 (Transfer Learning)")

# You can choose which model to use
MODEL_CHOICE = "improved_cnn"  # Options: "improved_cnn", "resnet50", "efficientnet"

torch.manual_seed(42)

if MODEL_CHOICE == "improved_cnn":
    model = ImprovedWildfireCNN(num_classes=len(class_names), dropout_rate=Config.DROPOUT_RATE)
    print("✓ Using Improved Custom CNN")
elif MODEL_CHOICE == "resnet50":
    model = create_resnet_model(num_classes=len(class_names), pretrained=True)
    print("✓ Using ResNet50 with Transfer Learning")
elif MODEL_CHOICE == "efficientnet":
    model = create_efficientnet_model(num_classes=len(class_names), pretrained=True)
    print("✓ Using EfficientNet-B0 with Transfer Learning")
else:
    model = ImprovedWildfireCNN(num_classes=len(class_names))
    print("✓ Using Improved Custom CNN (default)")

# Move model to device
model = model.to(device)
print(f"Model moved to device: {device}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Print model architecture (first few layers only to avoid clutter)
print("\nModel Architecture Preview:")
print(str(model)[:500] + "..." if len(str(model)) > 500 else str(model))

# Enhanced Training Setup and Execution
print("\n=== Training Configuration ===")
print(f"Epochs: {Config.EPOCHS}")
print(f"Learning Rate: {Config.LEARNING_RATE}")
print(f"Batch Size: {Config.BATCH_SIZE}")
print(f"Image Size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
print(f"Weight Decay: {Config.WEIGHT_DECAY}")

# Train the model (only if we have data)
if train_dataloader is not None and test_dataloader is not None:
    torch.manual_seed(42)
    
    print(f"\n=== Starting Enhanced Training ===")
    print("Features of enhanced training:")
    print("- AdamW optimizer with weight decay")
    print("- Cosine annealing learning rate scheduler")
    print("- Early stopping with patience")
    print("- Gradient clipping")
    print("- Advanced metrics (F1-score)")
    
    # Use the enhanced training function from improved_model.py
    results = train_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=Config.EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        device=device
    )
    
    print(f"\n=== Training Results ===")
    if results:
        best_acc = max(results['test_acc'])
        print(f"Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        
        # Plot training curves
        print("\nGenerating training visualization...")
        plot_training_curves(results, save_path="training_curves.png")
        
        # Get final predictions for confusion matrix
        model.eval()
        y_true, y_pred = [], []
        with torch.inference_mode():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                predictions = model(X)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(predictions.argmax(1).cpu().numpy())
        
        # Plot confusion matrix
        plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png")
        
        print("✓ Training visualization saved as 'training_curves.png'")
        print("✓ Confusion matrix saved as 'confusion_matrix.png'")
    
else:
    print("Cannot train model: No data available.")
    print("Please ensure the wildfire dataset is properly placed in the data directory.")
    results = None

# Model Comparison (if you want to compare different architectures)
print("\n=== Model Comparison Option ===")
print("To compare different model architectures, you can:")
print("1. Change MODEL_CHOICE variable to 'resnet50' or 'efficientnet'")
print("2. Re-run the training section")
print("3. Compare the results")

print("\n=== Summary of Improvements ===")
print("✓ Enhanced data augmentation (rotation, color jitter, gaussian blur)")
print("✓ Improved CNN architecture with batch normalization")
print("✓ Transfer learning options (ResNet50, EfficientNet)")
print("✓ Advanced training techniques (early stopping, LR scheduling)")
print("✓ Better evaluation metrics and visualization")
print("✓ Gradient clipping and weight decay regularization")
print("\nThese improvements should significantly boost your model accuracy!")