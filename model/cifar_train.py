import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import sys
import os

# Add current and parent directories to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from resnet_unet import ResNetUNet, CIFAR10Subset
from loss.focal_loss import FocalLoss
from loss.focal_dice_loss import Focal_Dice_Loss


def train_model_focal(model, train_loader, val_loader, num_epochs=50):
    """
    Applies a training loop using Focal Loss on CIFAR10 dataset
    """
    # Set device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Define FocalLoss with weights for 4 classes
    alpha = [1.0, 1.0, 1.0, 1.0]  # Equal weights for all classes
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
        # Training Phase
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
        
        # Validation Phase
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
        print(f'Training Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
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
            }, 'best_model_cifar_focal.pth')
            print(f"Model saved with validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
                
    return history

def train_model_focal_dice(model, train_loader, val_loader, num_epochs=10):
    """
    Applies a training loop using Focal-Dice Loss on CIFAR10 dataset
    """
    # Set device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Define Focal-Dice Loss - pass alpha as a list instead of a tensor
    alpha = [0.5, 0.5, 0.5, 0.5]  # Pass as regular Python list, not tensor
    criterion = Focal_Dice_Loss(gamma=2, alpha=alpha, weights={'focal': 0.5, 'dice': 0.5})
    # Move criterion to the same device as the model
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
    
    # Training loop
    for epoch in range(num_epochs):
        # Training Phase
        model.train()  
        train_loss = 0  
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) 
            
            optimizer.zero_grad()  
            outputs = model(images) 
            
            if len(outputs.shape) == 4:  # Handle UNet output (B, C, H, W)
                # Average over spatial dimensions for focal loss calculation
                outputs_for_loss = outputs.mean([2, 3])  # Reduce to [B, C]
                
                # Now pass the spatially-reduced outputs to the loss function
                loss = criterion(outputs_for_loss, labels)
                
                # For accuracy calculation, use the same reduced output
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
        
        # Validation Phase
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
        print(f'Training Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
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
            }, 'best_model_cifar_focal_dice.pth')
            print(f"Model saved with validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
                
    return history

def main():
    # Set data paths and parameters
    data_dir = "./data"
    batch_size = 64
    num_epochs = 50
    
    # The class names we'll use (first 4 CIFAR10 classes)
    classes = [0, 1, 2, 3]  # airplane, automobile, bird, cat
    
    # Define image transformations for CIFAR10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Create training and validation datasets
    print("Loading CIFAR10 dataset...")
    train_dataset = CIFAR10Subset(root=data_dir, train=True, transform=transform_train, classes=classes)
    val_dataset = CIFAR10Subset(root=data_dir, train=False, transform=transform_test, classes=classes)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model
    print("Initializing ResNetUNet model...")
    model = ResNetUNet(n_classes=4)  # 4 classes

    # Choose which loss function to use
    loss_function = "focal_dice"  # Options: "focal" or "focal_dice"
    
    if loss_function == "focal":
        print("Training with Focal Loss...")
        history = train_model_focal(model, train_loader, val_loader, num_epochs=num_epochs)
        print("Training complete. Best model saved as 'best_model_cifar_focal.pth'.")
    elif loss_function == "focal_dice":
        print("Training with Focal-Dice Loss...")
        history = train_model_focal_dice(model, train_loader, val_loader, num_epochs=num_epochs)
        print("Training complete. Best model saved as 'best_model_cifar_focal_dice.pth'.")
    else:
        print("Invalid loss function specified. Choose 'focal' or 'focal_dice'.")

if __name__ == "__main__":
    main()
