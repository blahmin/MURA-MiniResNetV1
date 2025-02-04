import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from model import MiniResNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

# MURA Dataset Class
class MURADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Traverse dataset and load all images
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
                    label = 1 if 'positive' in root.lower() else 0  # 1 = fracture, 0 = no fracture
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 32
epochs = 100
learning_rate = 0.0003
num_classes = 2

# Data directories
train_dir = r"D:\MURA-v1.1\MURA-v1.1\train"  # Contains all body parts
val_dir = r"D:\MURA-v1.1\MURA-v1.1\valid"

# Data Transforms

transform = transforms.Compose([
    # Step 1: Random Affine Transform (rotation, translation, scaling)
    transforms.RandomAffine(
        degrees=10,  # Rotate up to ±10 degrees
        translate=(0.05, 0.05),  # Shift horizontally/vertically by up to 5%
        scale=(0.9, 1.1),  # Scale the image between 90% and 110%
    ),
    
    # Step 2: Elastic Transform (simulated tissue/bone deformation)
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)  # Mimics elastic effects with slight blurring
    ], p=0.5),  # Applied with 50% probability
    
    # Step 4: Random Perspective Transform (simulates imperfect angles)
    transforms.RandomPerspective(distortion_scale=0.1, p=0.3),  # 30% chance of perspective distortion

    # Basic Augmentations
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
    transforms.Resize((224, 224)),  # Resize to input size for the CNN
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize (ImageNet-like normalization)
])
# Datasets and DataLoaders
train_dataset = MURADataset(train_dir, transform=transform)
val_dataset = MURADataset(val_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model Initialization
model = MiniResNet(num_classes=num_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # Halves LR every 10 epochs
early_stopping_patience = 10  # Number of epochs to wait for improvement
best_val_acc = 0
patience_counter = 0

# Training Function
def train():
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}]: Train Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f'Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # Save the best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")

        # Step the scheduler
        scheduler.step()

        print("------------------------------------------------------------")

# Main Function
if __name__ == '__main__':
    print("Starting Training...")
    train()
