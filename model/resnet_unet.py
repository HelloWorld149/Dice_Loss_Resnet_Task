import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW

# Import loss functions
from loss.focal_loss import FocalLoss
from loss.focal_dice_loss import DiceLoss, Focal_Dice_Loss

# double conv function
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Conv2D
            nn.BatchNorm2d(out_channels),                                    # Batch normalization
            nn.ReLU(inplace=True),                                           #  (ReLU)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # Conv2D
            nn.BatchNorm2d(out_channels),                                    # Batch normalization
            nn.ReLU(inplace=True)                                         
        )

    def forward(self, x):
       return self.double_conv(x)
    
# SCSE (Spatial and Channel Squeeze & Excitation) modülü
class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
       
        super().__init__()
        # Kanal Squeeze ve Excitation (cSE) Bloğu
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                      
            nn.Conv2d(in_channels, in_channels // reduction, 1), 
            nn.ReLU(inplace=True),                       
            nn.Conv2d(in_channels // reduction, in_channels, 1),  
            nn.Sigmoid()                                  
        )
        # Uzamsal Squeeze ve Excitation (sSE) Bloğu
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),                  
            nn.Sigmoid()                               
        )

    def forward(self, x):
       
       
        return x * self.cSE(x) + x * self.sSE(x)

# ResNet-based UNet architecture is defined
class ResNetUNet(nn.Module):
    def __init__(self, n_classes=1):
        """ Defines the ResNet-50 based UNet model"""
        super().__init__()
        
        # Load ResNet-50 with pre-trained weights
        self.base_model = models.resnet50(pretrained=True)

        # Encoder1: Get the first convolution layers of ResNet
        self.encoder1 = nn.Sequential(
            self.base_model.conv1, 
            self.base_model.bn1,    
            self.base_model.relu    
        )
        self.pool = self.base_model.maxpool  

        # ResNet layers are defined as Encoders
        self.encoder2 = self.base_model.layer1 
        self.encoder3 = self.base_model.layer2  
        self.encoder4 = self.base_model.layer3 

        # Upsampling layers and DoubleConv modules for Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) 
        self.decoder4 = DoubleConv(1024, 512)  

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  
        self.decoder3 = DoubleConv(512, 256)  

        self.upconv2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)  
        self.decoder2 = DoubleConv(128, 64)  

        # Final upsampling and convolution layers to restore resolution to original size
        self.upconv_final = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),                         
            nn.ReLU(inplace=True),                      
            nn.Conv2d(32, n_classes, kernel_size=1),    
            nn.Sigmoid()                                 
        )

        self.scse = SCSEModule(64)

    def forward(self, x):
        """ Forward data flow: Transfer data from input to output. """
        # Feature extraction from Encoder
        enc1 = self.encoder1(x)        
        enc1 = self.scse(enc1)         
        enc2 = self.encoder2(self.pool(enc1))  
        enc3 = self.encoder3(enc2)   
        enc4 = self.encoder4(enc3)    

        # Decoder stage: Restore feature map to original size
        dec4 = self.upconv4(enc4)                      
        dec4 = torch.cat((dec4, enc3), dim=1)           
        dec4 = self.decoder4(dec4)                   

        dec3 = self.upconv3(dec4)                     
        dec3 = torch.cat((dec3, enc2), dim=1)          
        dec3 = self.decoder3(dec3)                    

        dec2 = self.upconv2(dec3)                       
        dec2 = torch.cat((dec2, enc1), dim=1)           
        dec2 = self.decoder2(dec2)                     

       
        x = self.upconv_final(dec2)                   
        x = self.final_conv(x)                         

        return x  

    


# Training function using focal loss
def train_model_focal(model, train_loader, val_loader, num_epochs=50):
    """Applies a training loop to train and validate the model."""
    # Set device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Transfer model to device
    
    # Define FocalLoss as the loss function
    criterion = FocalLoss(alpha=1, gamma=2)
    
    # Define AdamW optimization algorithm
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Define learning rate scheduler (OneCycleLR)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Variables to track best validation loss
    best_val_loss = float('inf')
    patience = 10 
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # **Training Phase**
        model.train()  
        train_loss = 0  

        # Iterate over training dataset
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device) 
            
            optimizer.zero_grad()  
            outputs = model(images) 
            loss = criterion(outputs, masks.unsqueeze(1))  
            loss.backward()  
            
            # Prevent gradient explosion by clipping gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()  
            scheduler.step()  
            
            train_loss += loss.item()  
        
        # **Validation Phase**
        model.eval()  
        val_loss = 0
        val_dice = 0
        val_iou = 0

        # Iterate over validation dataset (no gradients)
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks.unsqueeze(1)).item()
                val_dice += dice_coefficient(outputs, masks.unsqueeze(1)).item()
                val_iou += iou_score(outputs, masks.unsqueeze(1)).item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        
        # Print training and validation results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Dice: {val_dice:.4f}')
        print(f'Validation IoU: {val_iou:.4f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model.pth')
        else:
            patience_counter += 1  # Increment counter for early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break

# Training function using focal loss + dice loss
def train_model_focal_dice(model, train_loader, val_loader, num_epochs=50):
    """Applies a training loop to train and validate the model using Focal-Dice loss."""
    # Set device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Transfer model to device
    
    # Define Focal-Dice Loss as the loss function
    criterion = Focal_Dice_Loss(gamma=2, alpha=0.5, weights={'focal': 0.5, 'dice': 0.5})
    
    # Define AdamW optimization algorithm
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Define learning rate scheduler (OneCycleLR)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Variables to track best validation loss
    best_val_loss = float('inf')
    patience = 10 
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # **Training Phase**
        model.train()  
        train_loss = 0  

        # Iterate over training dataset
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device) 
            
            optimizer.zero_grad()  
            outputs = model(images) 
            loss = criterion(outputs, masks.unsqueeze(1))  
            loss.backward()  
            
            # Prevent gradient explosion by clipping gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()  
            scheduler.step()  
            
            train_loss += loss.item()  
        
        # **Validation Phase**
        model.eval()  
        val_loss = 0
        val_dice = 0
        val_iou = 0

        # Iterate over validation dataset (no gradients)
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks.unsqueeze(1)).item()
                val_dice += dice_coefficient(outputs, masks.unsqueeze(1)).item()
                val_iou += iou_score(outputs, masks.unsqueeze(1)).item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        
        # Print training and validation results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Dice: {val_dice:.4f}')
        print(f'Validation IoU: {val_iou:.4f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model_focal_dice.pth')
        else:
            patience_counter += 1  # Increment counter for early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break

# Dice coefficient calculation function for evaluation
def dice_coefficient(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# IoU score calculation function for evaluation
def iou_score(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# Main execution function
if __name__ == "__main__":
    # Set training data paths
    train_data_dir = "/root/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/train_set"
    train_label_files = [
        os.path.join(train_data_dir, "label_data_0313.json"),
        os.path.join(train_data_dir, "label_data_0531.json")
    ]
    val_label_file = os.path.join(train_data_dir, "label_data_0601.json")

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image dimensions to 224x224
        transforms.ToTensor(),         # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Create training and validation datasets
    train_dataset = TuSimpleDataset(train_data_dir, train_label_files, transform=transform, augment=True)
    val_dataset = TuSimpleDataset(train_data_dir, [val_label_file], transform=transform, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Initialize model
    model = ResNetUNet(n_classes=1)

    # Choose which loss function to use
    loss_function = "focal_dice"  # Options: "focal" or "focal_dice"
    num_epochs = 50
    
    if loss_function == "focal":
        print("Training with Focal Loss...")
        train_model_focal(model, train_loader, val_loader, num_epochs=num_epochs)
        print("Training complete. Best model saved as 'best_model.pth'.")
    elif loss_function == "focal_dice":
        print("Training with Focal-Dice Loss...")
        train_model_focal_dice(model, train_loader, val_loader, num_epochs=num_epochs)
        print("Training complete. Best model saved as 'best_model_focal_dice.pth'.")
    else:
        print("Invalid loss function specified. Choose 'focal' or 'focal_dice'.")