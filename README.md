# Focal-Dice Loss ResNet Task

## Attribution
- The implementation of `focal_loss` and `dice_loss` used in this project are from [ShannonAI/dice_loss_for_NLP](https://github.com/ShannonAI/dice_loss_for_NLP).
- The ResNet UNet implementation (`resnet_unet.py`) is adapted from [YusufAtti/TuSimple-Lane-Detection-with-ResNet-UNet-Focal-Loss](https://github.com/YusufAtti/TuSimple-Lane-Detection-with-ResNet-UNet-Focal-Loss/blob/main/resnet_unet.py).

## Project Overview
This project implements a ResNet model using Dice Loss + focal loss for image classification tasks on the CIFAR10 dataset. By combining these loss functions, the model achieves better handling of class imbalance and improved performance on difficult-to-classify examples.

## Project Structure
```
Dice_Loss_Resnet_Task/
│
├── model/
│   ├── cifar_train.py       # Main training script for CIFAR10 dataset
│   └── resnet_unet.py       # ResNetUNet model implementation
│
├── loss/
│   ├── focal_loss.py        # Focal Loss implementation
│   └── focal_dice_loss.py   # Combined Focal and Dice Loss implementation
│
├── data/                    # Directory for storing CIFAR10 dataset
│
├── best_model_cifar_focal.pth       # Saved model with Focal Loss
└── best_model_cifar_focal_dice.pth  # Saved model with Focal-Dice Loss
```

## Model Architecture
The `ResNetUNet` architecture combines a ResNet backbone (typically ResNet50) with U-Net style skip connections. This architecture provides:
- Strong feature extraction from the ResNet backbone
- Detailed spatial information preservation through skip connections
- Effective classification capability for image data

## Loss Functions
This project implements and compares two loss functions:

### 1. Focal Loss
- Addresses class imbalance by down-weighting easy examples
- Focuses training on hard negative examples
- Parameters:
  - `gamma`: Focusing parameter (default: 2)
  - `alpha`: Class weights for balancing

### 2. Focal-Dice Loss
- Combines Focal Loss with Dice Loss for improved performance
- Dice Loss component measures overlap between predictions and ground truth
- Parameters:
  - `gamma`: Focal Loss focusing parameter
  - `alpha`: Class weights
  - `weights`: Relative weighting between Focal and Dice components

## Usage
To train a model on CIFAR10 dataset:

```bash
python model/cifar_train.py
```

You can modify the training parameters by editing the `main()` function in `cifar_train.py`. Choose between Focal Loss and Focal-Dice Loss by changing the `loss_function` variable.

## Requirements
- PyTorch >= 1.7.0
- torchvision
- numpy

## Results
The model achieves improved accuracy on four classes of the CIFAR10 classification tasks, especially with the combined Focal-Dice loss function. Early experiments show that the Focal-Dice loss helps the model converge faster and achieve better validation accuracy compared to focal loss.
