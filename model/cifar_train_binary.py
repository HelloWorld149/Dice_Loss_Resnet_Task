import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import numpy as np
import sys
import os
from torchvision.datasets import CIFAR10
import warnings


from sklearn.metrics import f1_score, matthews_corrcoef

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Add current and parent directories to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from resnet_unet import ResNetUNet
from loss.focal_loss import FocalLoss
from loss.focal_dice_loss import Focal_Dice_Loss


# Custom CIFAR10 subset with only selected classes and a desired imbalance ratio
class CIFAR10Subset(Dataset):
    def __init__(self, root, train=True, transform=None, download=True, classes=[3, 5], ratio=None):
        """
        CIFAR10 dataset with only selected classes.
        Args:
            root: Root directory for the dataset.
            train: If True, use training set, otherwise use test set.
            transform: Transformations to apply to the images.
            download: If True, download the dataset.
            classes: List of class indices to include. (Default: [3, 5] for cat and dog.)
            ratio: If provided and classes == [3,5], force cat:dog ratio of 1:ratio.
                   For example, ratio=20 means retain 1 cat for every 20 dog images.
        """
        self.full_dataset = CIFAR10(root=root, train=train, transform=transform, download=download)
        self.class_indices = classes
        
        # First, collect all indices of the selected classes.
        self.indices = [i for i, (_, label) in enumerate(self.full_dataset) if label in self.class_indices]
        
        # If a ratio is specified and the selected classes are cat (3) and dog (5), adjust the dataset.
        if ratio is not None and set(self.class_indices) == {3, 5}:
            # Separate indices for cat and dog.
            cat_indices = [i for i in self.indices if self.full_dataset[i][1] == 3]
            dog_indices = [i for i in self.indices if self.full_dataset[i][1] == 5]
            # Subsample cat indices: target number is (number of dog images) / ratio.
            desired_cat_count = len(dog_indices) // ratio
            if len(cat_indices) > desired_cat_count:
                cat_indices = np.random.choice(cat_indices, desired_cat_count, replace=False).tolist()
            # Combine the subsampled cat indices with all dog indices.
            self.indices = cat_indices + dog_indices
            # Shuffle the indices for randomness.
            np.random.shuffle(self.indices)
        
        # Create a mapping from old labels to new labels (0 to n_classes-1)
        self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(self.class_indices)}
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image, label = self.full_dataset[self.indices[idx]]
        new_label = self.label_mapping[label]
        return image, new_label


def train_model_focal(model, train_loader, val_loader, num_epochs=10):
    """
    Training loop using Focal Loss on CIFAR10 dataset.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    alpha = [0.5, 0.5]  
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
    best_val_acc = 0
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
            
            # Handle potential UNet output
            if len(outputs.shape) == 4:
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
        
        if val_loss < best_val_loss:
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
    alpha = [0.5, 0.5]  
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
        if val_loss < best_val_loss:
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



def evaluate_model(model_path, test_loader, n_classes=2):
    """
    Evaluates a saved model on the test dataset.
    Computes overall accuracy, per-class accuracy, F1 score, and Matthews Correlation Coefficient (MCC).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetUNet(n_classes=n_classes)
    model.to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    class_correct = [0] * n_classes
    class_total = [0] * n_classes

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
            
            # Accumulate per-sample labels for overall metrics.
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            # Compute per-class counts
            for i in range(labels.size(0)):
                lab = labels[i].item()
                class_total[lab] += 1
                if predicted[i].item() == lab:
                    class_correct[lab] += 1
    
    test_acc = 100 * test_correct / test_total
    class_acc = [100 * c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]
    
    # Compute the weighted F1 score and the Matthews Correlation Coefficient.
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_predictions)
    
    print(f"\nTest Evaluation Results for {model_path}:")
    print(f"Overall Test Accuracy: {test_acc:.2f}%")
    print("Per-class Test Accuracy:")
    class_names = ['cat', 'dog']  # Assuming new labels: 0 for cat, 1 for dog.
    for i, acc in enumerate(class_acc):
        print(f"  {class_names[i]}: {acc:.2f}%")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    
    return test_acc, class_acc, f1, mcc


def main():
    data_dir = "./data"
    batch_size = 64
    num_epochs = 10
    
    target_ratio = 1  
    
    # Use only classes [3, 5]: cat (3) and dog (5)
    classes = [3, 5]
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print("Loading CIFAR10 dataset...")
    full_train_dataset = CIFAR10Subset(root=data_dir, train=True, transform=transform_train,
                                       download=True, classes=classes, ratio=target_ratio)
    test_dataset = CIFAR10Subset(root=data_dir, train=False, transform=transform_test,
                                 download=True, classes=classes, ratio=target_ratio)
    
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
    model = ResNetUNet(n_classes=2)  # 2 classes: cat and dog
    
    # Train using Focal Loss.
    history = train_model_focal_dice(model, train_loader, val_loader, num_epochs=num_epochs)
    print("Training complete. Best model saved as 'best_model_cifar_focal_dice.pth'.")
    evaluate_model('best_model_cifar_focal_dice.pth', test_loader)

    history = train_model_focal(model, train_loader, val_loader, num_epochs=num_epochs)
    print("Training complete. Best model saved as 'best_model_cifar_focal.pth'.")
    evaluate_model('best_model_cifar_focal.pth', test_loader)
    

if __name__ == "__main__":
    main()
