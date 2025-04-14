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
import warnings

from sklearn.metrics import f1_score, matthews_corrcoef

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Add current and parent directories to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from resnet_unet import ResNetUNet, CIFAR10Subset
from loss.focal_loss import FocalLoss
from loss.focal_dice_loss import Focal_Dice_Loss


def train_model_focal(model, train_loader, val_loader, num_epochs=10):
    """
    Applies a training loop using Focal Loss on CIFAR10 dataset.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Define FocalLoss with weights for 4 classes - giving more weight to minority classes
    alpha = [10.0, 10.0, 10.0, 1.0]  # Higher weights for minority classes (assuming class 3 is majority)
    criterion = FocalLoss(gamma=2, alpha=alpha, reduction="mean")
    if hasattr(criterion, 'alpha') and criterion.alpha is not None:
        criterion.alpha = criterion.alpha.to(device)
    
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    best_val_loss = float('inf')
    patience = 10 
    patience_counter = 0
    
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
                outputs = outputs.mean([2, 3])
            
            loss = criterion(outputs, labels)  
            loss.backward()  
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  
            scheduler.step()  
            
            train_loss += loss.item()            
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        model.eval()  
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if len(outputs.shape) == 4:
                    outputs = outputs.mean([2, 3])
                    
                val_loss += criterion(outputs, labels).item()                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        if val_loss < best_val_loss:  # Save based on best validation loss
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'accuracy': val_acc,
            }, 'best_model_cifar_focal.pth')
            print(f"Model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
    return history

def train_model_focal_dice(model, train_loader, val_loader, num_epochs=10, checkpoint_path=None, continue_epochs=5):
    """
    Applies a training loop using Focal-Dice Loss on CIFAR10 dataset.
    Can continue training from a checkpoint if provided.
    
    Args:
        continue_epochs: Number of additional epochs to train when resuming from checkpoint
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Reduce the weight disparity to improve majority class performance
    # Current weights [10.0, 10.0, 10.0, 1.0] may be too extreme
    alpha = [5.0, 5.0, 5.0, 2.0]  # More balanced weights that still prioritize minority classes
    criterion = Focal_Dice_Loss(gamma=2, alpha=alpha, weights={'focal': 0.5, 'dice': 0.5})
    if hasattr(criterion, 'alpha') and criterion.alpha is not None:
        criterion.alpha = criterion.alpha.to(device)
    
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    # Load from checkpoint if provided
    best_val_loss = float('inf')
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch'] + 1
        # Adjust num_epochs to ensure we train for additional epochs beyond the checkpoint
        num_epochs = start_epoch + continue_epochs
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with validation loss: {best_val_loss:.4f}")
        print(f"Will continue training for {continue_epochs} more epochs (until epoch {num_epochs})")
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    patience = 10 
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(start_epoch, num_epochs):
        model.train()  
        train_loss = 0  
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)             
            optimizer.zero_grad()  
            outputs = model(images) 
            
            if len(outputs.shape) == 4:
                outputs_for_loss = outputs.mean([2, 3])
                loss = criterion(outputs_for_loss, labels)
                outputs_for_acc = outputs_for_loss
            else:
                loss = criterion(outputs, labels)
                outputs_for_acc = outputs

            loss.backward()  
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  
            scheduler.step()  
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs_for_acc, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        model.eval()  
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if len(outputs.shape) == 4:
                    outputs_for_loss = outputs.mean([2, 3])
                    val_loss += criterion(outputs_for_loss, labels).item()
                    outputs_for_acc = outputs_for_loss
                else:
                    val_loss += criterion(outputs, labels).item()
                    outputs_for_acc = outputs
                
                _, predicted = torch.max(outputs_for_acc, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        if val_loss < best_val_loss:  # Save based on best validation loss
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'accuracy': val_acc,
            }, 'best_model_cifar_focal_dice.pth')
            print(f"Model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
    return history

def evaluate_model(model_path, test_loader, n_classes=4):
    """
    Evaluates a saved model on the test dataset.
    Computes overall accuracy, per-class accuracy, weighted F1 score, and Matthews Correlation Coefficient (MCC).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetUNet(n_classes=n_classes)
    model.to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    class_correct = [0] * n_classes
    class_total = [0] * n_classes

    # Lists to accumulate all labels and predictions
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if len(outputs.shape) == 4:
                outputs = outputs.mean([2, 3])
            _, predicted = torch.max(outputs, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Accumulate for F1 and MCC
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
    
    test_acc = 100 * test_correct / test_total
    class_acc = [100 * correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
    
    # Compute weighted F1 score and MCC using scikit-learn
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_predictions)
    
    print(f"\nTest Evaluation Results for {model_path}:")
    print(f"Overall Test Accuracy: {test_acc:.2f}%")
    print("Per-class Test Accuracy:")
    class_names = ['airplane', 'automobile', 'bird', 'cat']
    for i, acc in enumerate(class_acc):
        print(f"  {class_names[i]}: {acc:.2f}%")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    
    return test_acc, class_acc, f1, mcc

def main():
    data_dir = "./data"
    batch_size = 64
    num_epochs = 5
    
    # The class names we'll use (first 4 CIFAR10 classes)
    classes = [0, 1, 2, 3]  # airplane, automobile, bird, cat
    
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

    print("Loading CIFAR10 dataset...")
    full_train_dataset = CIFAR10Subset(root=data_dir, train=True, transform=transform_train, classes=classes)
    test_dataset = CIFAR10Subset(root=data_dir, train=False, transform=transform_test, classes=classes)
    
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Initializing ResNetUNet model...")
    #model = ResNetUNet(n_classes=4)  # 4 classes

    # Train with Focal Loss
    #print("\n=== Training with Focal Loss ===")
    #focal_model = ResNetUNet(n_classes=4)
    #train_model_focal(focal_model, train_loader, val_loader, num_epochs=num_epochs)
    
    # Train with Focal-Dice Loss from checkpoint if exists
    print("\n=== Training with Focal-Dice Loss ===")
    #focal_dice_model = ResNetUNet(n_classes=4)
    #train_model_focal_dice(focal_dice_model, train_loader, val_loader, num_epochs=num_epochs, 
    #                       checkpoint_path='best_model_cifar_focal_dice.pth', continue_epochs=5)

    # Evaluate models saved from training with Focal Loss and Focal-Dice Loss
    print("\n=== Evaluating Models ===")
    #evaluate_model('best_model_cifar_focal.pth', test_loader)
    evaluate_model('best_model_cifar_focal_dice.pth', test_loader)

if __name__ == "__main__":
    main()
