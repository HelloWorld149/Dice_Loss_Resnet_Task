import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from torchvision.datasets import CIFAR10

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
    def __init__(self, n_classes=4):  # Changed to 4 classes for CIFAR10 subset
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
            # No sigmoid for multi-class classification
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

        return x  # Returns logits for multi-class classification

# Custom CIFAR10 subset with only 4 classes
class CIFAR10Subset(Dataset):
    def __init__(self, root, train=True, transform=None, download=True, classes=[0, 1, 2, 3]):
        """
        CIFAR10 dataset with only selected classes (default: first 4 classes)
        Args:
            root: Root directory for the dataset
            train: If True, use training set, otherwise use test set
            transform: Transformations to apply to the images
            download: If True, download the dataset
            classes: List of class indices to include (0-9)
        """
        self.full_dataset = CIFAR10(root=root, train=train, transform=transform, download=download)
        self.class_indices = classes
        
        # Filter indices belonging to the selected classes
        self.indices = [i for i, (_, label) in enumerate(self.full_dataset) if label in self.class_indices]
        
        # Create a mapping from old labels to new labels (0 to n_classes-1)
        self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(self.class_indices)}
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image, label = self.full_dataset[self.indices[idx]]
        # Map the original label to the new label range (0 to n_classes-1)
        new_label = self.label_mapping[label]
        return image, new_label