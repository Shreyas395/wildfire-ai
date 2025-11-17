# Early Wildfire Detection Model 
import numpy as np
import pandas as pd
import os
from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from typing import List
from datetime import datetime
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

CLASS_NAMES = ['fire', 'nofire']
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def is_image_file(path: Path) -> bool:
    """Return True if the path points to a valid image file."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def get_image_files(directory: Path) -> List[Path]:
    """Return all image files inside a directory (recursively)."""
    if not directory.exists():
        return []
    return [file for file in directory.rglob('*') if is_image_file(file)]


def count_images(directory: Path) -> int:
    """Count image files inside a directory."""
    return len(get_image_files(directory))


def get_image_paths(data_dir: Path, class_names: List[str] | None = None) -> tuple[list, list]:
    """Get image paths and corresponding labels."""
    class_names = class_names or CLASS_NAMES
    image_paths: list[Path] = []
    labels: list[int] = []

    for class_index, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory {class_dir} not found.")
            continue

        class_images = [path for path in class_dir.iterdir() if is_image_file(path)]
        if not class_images:
            print(f"Warning: No image files found in {class_dir}.")
            continue

        image_paths.extend(class_images)
        labels.extend([class_index] * len(class_images))

    return image_paths, labels


def validate_dataset_structure(dirs: dict, class_names: List[str] = None) -> tuple[list, list]:
    """Return missing and empty directories for the dataset."""
    class_names = class_names or CLASS_NAMES
    missing_dirs, empty_dirs = [], []

    for split in ['train', 'val', 'test']:
        split_dir = dirs.get(split)
        for class_name in class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                missing_dirs.append(class_dir)
            elif count_images(class_dir) == 0:
                empty_dirs.append(class_dir)

    return missing_dirs, empty_dirs


def gather_all_image_paths(dirs: dict, class_names: List[str] = None) -> List[Path]:
    """Collect all image paths across dataset splits."""
    class_names = class_names or CLASS_NAMES
    collected: List[Path] = []
    for split in ['train', 'val', 'test']:
        split_paths, _ = get_image_paths(dirs[split], class_names)
        collected.extend(split_paths)
    return collected


def clean_unexpected_class_dirs(dirs: dict, allowed_class_names: List[str] = None) -> None:
    """Remove empty class directories that were auto-created but aren't expected."""
    allowed_class_names = allowed_class_names or CLASS_NAMES
    for split in ['train', 'val', 'test']:
        split_dir = dirs.get(split)
        if not split_dir or not split_dir.exists():
            continue

        for subdir in split_dir.iterdir():
            if not subdir.is_dir() or subdir.name in allowed_class_names:
                continue

            if count_images(subdir) == 0:
                try:
                    subdir.rmdir()
                    print(f"Removed unexpected empty directory: {subdir}")
                except OSError:
                    pass

def setup_data_directories(base_path: str = "data/the_wildfire_dataset_2n_version"):
    """Setup and verify the dataset directory structure."""
    data_path = Path(base_path)
    
    # Create directories if they don't exist
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    val_dir = data_path / "val"
    
    # Create class subdirectories
    for split_dir in [train_dir, test_dir, val_dir]:
        for class_name in CLASS_NAMES:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    return {
        'base': data_path,
        'train': train_dir,
        'test': test_dir,
        'val': val_dir
    }

# Setup data directories
paths = setup_data_directories()
train_dir = paths['train']
test_dir = paths['test']
val_dir = paths['val']

# Verify directory structure
def verify_structure(dirs: dict):
    """Verify the dataset directory structure."""
    print("\n=== Dataset Structure ===")
    for name, path in dirs.items():
        if not path.exists():
            print(f"Warning: {name} directory not found at {path}")
        else:
            print(f"\n{name.capitalize()} directory: {path}")
            for subdir in path.iterdir():
                if subdir.is_dir():
                    num_files = len(list(subdir.glob('*')))
                    print(f"  - {subdir.name}: {num_files} files")


# Create datasets and dataloaders
def create_dataloaders(
    train_dir: Path,
    test_dir: Path,
    val_dir: Path,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int | None = None
) -> tuple[DataLoader, DataLoader, DataLoader, list]:
    """Create training, validation, and test dataloaders."""
    # Define transforms
    train_transform = get_transforms('train')
    test_transform = get_transforms('test')

    # Create datasets
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)
    val_data = datasets.ImageFolder(root=val_dir, transform=test_transform)

    # Get class names
    class_names = train_data.classes

    pin_memory = device.type == "cuda"
    worker_count = num_workers if num_workers is not None else (Config.NUM_WORKERS if device.type == "cuda" else 0)
    persistent = pin_memory and worker_count > 0
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": worker_count,
        "pin_memory": pin_memory,
        "persistent_workers": persistent,
        "shuffle": True,
    }

    # Create dataloaders
    train_dataloader = DataLoader(train_data, **loader_kwargs)

    eval_loader_kwargs = loader_kwargs.copy()
    eval_loader_kwargs.update({"shuffle": False})

    test_dataloader = DataLoader(test_data, **eval_loader_kwargs)

    val_dataloader = DataLoader(val_data, **eval_loader_kwargs)

    return train_dataloader, val_dataloader, test_dataloader, class_names


def main():
    verify_structure(paths)
    clean_unexpected_class_dirs(paths)

    missing_dirs, empty_dirs = validate_dataset_structure(paths)
    if missing_dirs:
        print("\nWarning: The following required directories are missing:")
        for missing in missing_dirs:
            print(f" - {missing}")

    if empty_dirs:
        print("\nWarning: The following directories do not contain any image files:")
        for empty in empty_dirs:
            print(f" - {empty}")

    if missing_dirs or empty_dirs:
        print("\nPlease ensure each split contains 'fire' and 'nofire' images before training.")

    # Collect image paths for previews and plotting
    image_path_list = gather_all_image_paths(paths)

    # Inspect a random image if available
    if image_path_list:
        random_image_path = random.choice(image_path_list)
        img = Image.open(random_image_path)
        image_class = random_image_path.parent.stem
        print(f"\nSample image: {random_image_path}")
        print(f"Class: {image_class}")
        print(f"Dimensions: {img.size}")
    else:
        img = None
        print("\nNo images found across dataset splits. Please verify dataset contents.")

    # Set Random Seed
    random.seed(42)
    torch.manual_seed(42)

    # Create dataloaders
    train_dataloader = val_dataloader = test_dataloader = None
    class_names = CLASS_NAMES.copy()

    try:
        train_dataloader, val_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir=paths['train'],
            test_dir=paths['test'],
            val_dir=paths['val'],
            batch_size=Config.BATCH_SIZE
        )

        print(f"\n=== Dataset Information ===")
        print(f"Number of training batches: {len(train_dataloader)}")
        print(f"Number of validation batches: {len(val_dataloader)}")
        print(f"Number of test batches: {len(test_dataloader)}")
        print(f"Class names: {class_names}")

        # Get a batch of training data
        train_features, train_labels = next(iter(train_dataloader))
        print(f"\nFeature batch shape: {train_features.shape}")
        print(f"Labels batch shape: {train_labels.shape}")

    except Exception as e:
        print(f"Error creating dataloaders: {str(e)}")
        print("Please ensure the dataset is properly structured with 'train', 'val', and 'test' directories.")
        print("Each directory should contain 'fire' and 'nofire' subdirectories with corresponding images.")
        print("Dataloaders were not created; training will be skipped until the issue is resolved.")

    # Enhanced Data transformation pipeline
    print("\n=== Data Transformation Setup ===")
    train_transform = get_transforms('train')
    test_transform = get_transforms('test')

    print(f"Using image size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print("Enhanced augmentations applied for training data")

    # Test transform on sample image (if available)
    if 'img' in locals() and img is not None:
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

    # Use the earlier dataloaders for training validation
    if train_dataloader is not None:
        print(f"\n=== DataLoader Summary ===")
        print(f"Training batches: {len(train_dataloader)}")
        print(f"Validation batches: {len(val_dataloader)}")
        print(f"Test batches: {len(test_dataloader)}")
    else:
        print("No valid data available for DataLoader creation.")

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
            best_idx = int(np.argmax(results['test_acc']))
            best_acc = results['test_acc'][best_idx]
            best_loss = results['test_loss'][best_idx]
            best_train_acc = results['train_acc'][best_idx]
            print(f"Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%) at epoch {best_idx+1}")

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

            # Analyze metric trends for easier interpretation
            def describe_trend(name: str, values: list[float]) -> str:
                if not values:
                    return f"- {name}: no data collected"
                start, end = values[0], values[-1]
                direction = "decreased" if end < start else "increased"
                return f"- {name}: {direction} from {start:.4f} to {end:.4f}"

            print("\n=== Training Curve Analysis ===")
            print(describe_trend("Training loss", results.get('train_loss', [])))
            print(describe_trend("Validation loss", results.get('test_loss', [])))
            print(describe_trend("Training accuracy", results.get('train_acc', [])))
            print(describe_trend("Validation accuracy", results.get('test_acc', [])))

            # Persist best-performing model snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"wildfire_model_epoch{best_idx+1}_acc{best_acc:.3f}_{timestamp}.pth"
            torch.save({
                'epoch': best_idx + 1,
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'best_val_acc': best_acc,
                'best_val_loss': best_loss,
                'train_acc_at_best': best_train_acc,
                'config': {
                    'IMG_SIZE': Config.IMG_SIZE,
                    'BATCH_SIZE': Config.BATCH_SIZE,
                    'EPOCHS': Config.EPOCHS,
                    'LEARNING_RATE': Config.LEARNING_RATE
                }
            }, model_filename)
            print(f"✓ Best model checkpoint saved as '{model_filename}'")

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


if __name__ == "__main__":
    main()
