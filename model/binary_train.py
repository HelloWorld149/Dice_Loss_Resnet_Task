import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys
import os
import random
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from resnet_unet import ResNetUNet
from loss.focal_loss import FocalLoss
from loss.focal_dice_loss import DiceLoss, Focal_Dice_Loss

class ImbalancedCatDogDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, cat_dog_ratio=0.1, val_split=0.2):
        """
        Creates an imbalanced dataset of cats and dogs from the training data only.
        Parameters:
            root_dir: root directory of the dataset
            train: True for training split, False for validation split
            transform: transformations to be applied to the images
            cat_dog_ratio: the ratio of cats to dogs in the training set
            val_split: proportion of data to use for validation (0.0 to 1.0)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.cat_dog_ratio = cat_dog_ratio
        
        # Always use the training folder since test dataset doesn't have labels
        dataset_folder = os.path.join(root_dir, 'train')
        
        # Get all image paths
        self.cat_images = sorted([os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if 'cat' in f.lower()])
        self.dog_images = sorted([os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if 'dog' in f.lower()])
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Split cats and dogs into training and validation sets
        val_cats_size = int(len(self.cat_images) * val_split)
        val_dogs_size = int(len(self.dog_images) * val_split)
        
        # Shuffle the data before splitting
        random.shuffle(self.cat_images)
        random.shuffle(self.dog_images)
        
        if train:
            # Use the training part (80% by default)
            self.cat_images = self.cat_images[val_cats_size:]
            self.dog_images = self.dog_images[val_dogs_size:]
            
            # Apply imbalanced ratio for training set
            total_cats = len(self.cat_images)
            total_dogs = len(self.dog_images)
            
            # Ensure cat proportion is cat_dog_ratio / (1 + cat_dog_ratio)
            # For example, with ratio 0.1, cats should be 0.1/1.1 = ~9.09% of total
            proportion_cats = cat_dog_ratio / (1 + cat_dog_ratio)
            desired_total = total_cats / proportion_cats
            desired_dogs = desired_total - total_cats
            
            # If needed number of dogs is less than available, randomly select a subset
            if desired_dogs < total_dogs:
                self.dog_images = self.dog_images[:int(desired_dogs)]
            
            print(f"Training dataset: {len(self.cat_images)} cats and {len(self.dog_images)} dogs")
            print(f"cats : dogs = 1:{len(self.dog_images)/len(self.cat_images):.1f}")
        else:
            # Use the validation part (20% by default)
            self.cat_images = self.cat_images[:val_cats_size]
            self.dog_images = self.dog_images[:val_dogs_size]
            
            print(f"Validation dataset: {len(self.cat_images)} cats and {len(self.dog_images)} dogs")
            print(f"cats : dogs = 1:{len(self.dog_images)/len(self.cat_images):.1f}")
        
        # Merge cat and dog images, 0 for cat, 1 for dog
        self.image_paths = self.cat_images + self.dog_images
        self.labels = [0] * len(self.cat_images) + [1] * len(self.dog_images)
        
        # Shuffle the combined data
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)
        self.image_paths, self.labels = list(self.image_paths), list(self.labels)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_model_focal(model, train_loader, val_loader, num_epochs=50):
    """
    Apply training loop using Focal Loss on imbalanced cat/dog dataset
    """
    # Set device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Define FocalLoss for binary classification, with higher weight for minority class
    alpha = [0.9, 0.1]  # Set higher weight for cat (minority class)
    criterion = FocalLoss(gamma=2, alpha=alpha, reduction="mean")
    
    # Define AdamW optimization algorithm
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Define learning rate scheduler (OneCycleLR)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Variables to track best validation loss and accuracy
    best_val_loss = float('inf')
    best_val_acc = 0
    patience = 10 
    patience_counter = 0
    
    # History for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()  
        train_loss = 0  
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) 
            
            optimizer.zero_grad()  
            outputs = model(images) 
            
            if len(outputs.shape) == 4:  # Handle UNet output (B, C, H, W)
                # Average over spatial dimensions for classification
                outputs = outputs.mean([2, 3])
            
            loss = criterion(outputs, labels)  
            loss.backward()  
            
            # Prevent gradient explosion by clipping gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()  
            scheduler.step()  
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()  
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                if len(outputs.shape) == 4:  # Handle UNet output
                    outputs = outputs.mean([2, 3])
                    
                val_loss += criterion(outputs, labels).item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate average losses and accuracy
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print training and validation results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        
        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'accuracy': best_val_acc,
            }, 'best_model_binary_focal.pth')
            print(f"Model saved with validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
                
    return history

def train_model_focal_dice(model, train_loader, val_loader, num_epochs=50):
    """
    Use Focal-Dice Loss on imbalanced cat/dog dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    alpha = [0.9, 0.1]  # set higher weight for cat (minority class)
    criterion = Focal_Dice_Loss(gamma=2, alpha=alpha, weights={'focal': 0.5, 'dice': 0.5})
    
    # set alpha to device if it is not None
    if hasattr(criterion, 'alpha') and criterion.alpha is not None:
        criterion.alpha = criterion.alpha.to(device)
    
    # Define AdamW optimization algorithm
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Define learning rate scheduler (OneCycleLR)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Variables to track best validation loss and accuracy
    best_val_loss = float('inf')
    best_val_acc = 0
    patience = 10 
    patience_counter = 0
    
    # History for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        model.train()  
        train_loss = 0  
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) 
            
            optimizer.zero_grad()  
            outputs = model(images) 
            
            if len(outputs.shape) == 4:  # Handle UNet output (B, C, H, W)
                # Average over spatial dimensions for loss calculation
                outputs_for_loss = outputs.mean([2, 3])  # Reduce to [B, C]
                
                # Pass the spatially-reduced outputs to the loss function
                loss = criterion(outputs_for_loss, labels)
                
                # Use the same reduced output for accuracy calculation
                outputs_for_acc = outputs_for_loss
            else:
                # For standard classification output
                loss = criterion(outputs, labels)
                outputs_for_acc = outputs          

            loss.backward()  
            
            # Prevent gradient explosion by clipping gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()  
            scheduler.step()  
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs_for_acc, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()  
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                if len(outputs.shape) == 4:  # Handle UNet output
                    # Average over spatial dimensions
                    outputs_for_loss = outputs.mean([2, 3])
                    val_loss += criterion(outputs_for_loss, labels).item()
                    outputs_for_acc = outputs_for_loss
                else:
                    # For standard classification output
                    val_loss += criterion(outputs, labels).item()
                    outputs_for_acc = outputs
                
                # Calculate accuracy
                _, predicted = torch.max(outputs_for_acc, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate average losses and accuracy
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print training and validation results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        
        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'accuracy': best_val_acc,
            }, 'best_model_binary_focal_dice.pth')
            print(f"Model saved with validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
                
    return history

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a binary classifier with different loss functions')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-focal', action='store_true', help='Use Focal Loss')
    group.add_argument('-focal_dice', action='store_true', help='Use Focal-Dice Loss')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--ratio', type=float, default=0.1, help='Cat to dog ratio (e.g., 0.1 for 1:10)')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    data_dir = "../../data"
    batch_size = args.batch_size
    num_epochs = args.epochs
    cat_dog_ratio = args.ratio  # cat:dog = 1:10 (default)
    val_split = args.val_split  # Use 20% of training data for validation (default)
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading dataset...")
    train_dataset = ImbalancedCatDogDataset(
        root_dir=data_dir, 
        train=True, 
        transform=transform_train, 
        cat_dog_ratio=cat_dog_ratio,
        val_split=val_split
    )
    
    val_dataset = ImbalancedCatDogDataset(
        root_dir=data_dir, 
        train=False, 
        transform=transform_test,
        cat_dog_ratio=cat_dog_ratio,
        val_split=val_split
    )
    
    print(f"Training size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model
    print("Initializing ResNet...")
    model = ResNetUNet(n_classes=2)  # 2 classes for binary classification (cat/dog)

    # Determine loss function based on command line arguments
    if args.focal:
        print("Using Focal Loss...")
        history = train_model_focal(model, train_loader, val_loader, num_epochs=num_epochs)
        print("Finished training. The best model has been saved as: 'best_model_binary_focal.pth'")
    elif args.focal_dice:
        print("Using Focal-Dice Loss...")
        history = train_model_focal_dice(model, train_loader, val_loader, num_epochs=num_epochs)
        print("Finished training. The best model has been saved as: 'best_model_binary_focal_dice.pth'")

if __name__ == "__main__":
    main()
