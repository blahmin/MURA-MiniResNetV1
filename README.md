# MURA-MiniResNetV1 (Single Model)

## Overview  
MURA-MiniResNetV1 is a lightweight convolutional neural network inspired by ResNet, designed for binary classification of medical X-ray images, specifically abnormality vs no abnormlity in a bone x-ray. This is the first iteration of the model in the development process. It is built to be computationally efficient while incorporating residual connections to improve gradient flow and prevent vanishing gradients.

## Model Architecture  
- Initial convolutional layer (3 → 16) with batch normalization and ReLU activation  
- Three residual blocks:  
  - Block 1: (16 → 32) x2  
  - Block 2: (32 → 64) x2, with stride 2 for downsampling  
  - Block 3: (64 → 128) x2, with stride 2 for downsampling  
- Adaptive average pooling  
- Fully connected layer (128 → 2 classes)  

## Key Features  
- Residual learning through BasicBlock connections to stabilize training  
- Lightweight architecture compared to standard ResNet models  
- Adaptive average pooling to handle variable input sizes  
- Batch normalization for improved convergence  
- Data augmentation using Albumentations  

## Training Pipeline  
- Dataset: MURA (Musculoskeletal Radiographs)  
- Augmentations: Handled via Albumentations, supporting PIL-based transforms  
- Loss Function: CrossEntropyLoss  
- Optimizer: Adam  
- Learning Rate: 1e-4  
- Batch Size: Configurable via DataLoader  

## Installation  
Run the following command to install the required dependencies: pip install torch torchvision albumentations pillow

## Training the Model  
To train the model, you must download the stanford MURA datset here: https://aimi.stanford.edu/datasets/mura-msk-xrays
Then setup your own pathways for the datasets and set them up in line 50-51 in train.py

## The stage
This model represents the first iteration in a larger project focused on improving automated diagnosis using medical imaging. 
