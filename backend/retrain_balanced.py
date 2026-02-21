"""
Retrain the PyTorch model with better balance to fix the bias towards real predictions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import json
import random
import numpy as np

class BalancedDataset(Dataset):
    def __init__(self, data_dir):
        self.images = []
        self.labels = []
        
        # Load real images
        real_dir = os.path.join(data_dir, "real")
        real_images = []
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(real_dir, img_name)
                    real_images.append(img_path)
        
        # Load AI images
        ai_dir = os.path.join(data_dir, "ai")
        ai_images = []
        if os.path.exists(ai_dir):
            for img_name in os.listdir(ai_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(ai_dir, img_name)
                    ai_images.append(img_path)
        
        # Use equal numbers from each class
        min_count = min(len(real_images), len(ai_images))
        print(f"Found {len(real_images)} real images and {len(ai_images)} AI images")
        print(f"Using {min_count} images from each class for balanced training")
        
        # Randomly sample from each class
        real_sample = random.sample(real_images, min_count)
        ai_sample = random.sample(ai_images, min_count)
        
        # Add real images
        for img_path in real_sample:
            self.images.append(img_path)
            self.labels.append(1)  # Real = 1
        
        # Add AI images
        for img_path in ai_sample:
            self.images.append(img_path)
            self.labels.append(0)  # AI = 0
        
        print(f"Total dataset size: {len(self.images)}")
        print(f"AI images: {sum(1 for label in self.labels if label == 0)}")
        print(f"Real images: {sum(1 for label in self.labels if label == 1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize((224, 224))
            image = transforms.ToTensor()(image)
            return image, label
        except:
            return torch.zeros(3, 224, 224), label

def retrain_model():
    print("Retraining model with better balance...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Load balanced dataset
    dataset = BalancedDataset("dataset")
    
    # Debug: Check if the path exists
    import os
    print(f"Current working directory: {os.getcwd()}")
    print(f"Dataset path: {os.path.abspath('dataset')}")
    print(f"Dataset exists: {os.path.exists('dataset')}")
    if os.path.exists('dataset'):
        print(f"AI dir exists: {os.path.exists('dataset/ai')}")
        print(f"Real dir exists: {os.path.exists('dataset/real')}")
        if os.path.exists('dataset/ai'):
            ai_files = [f for f in os.listdir('dataset/ai') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"AI files found: {len(ai_files)}")
        if os.path.exists('dataset/real'):
            real_files = [f for f in os.listdir('dataset/real') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Real files found: {len(real_files)}")
    
    if len(dataset) == 0:
        print("No images found!")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Create model with simpler architecture to prevent overfitting
    model = models.resnet18(weights=None)
    model.npm run dev = nn.Sequential(
        nn.Dropout(0.3),  # Reduced dropout
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )
    
    # Training setup with class weights to balance the loss
    # Give more weight to AI class (0) to prevent bias towards real (1)
    class_weights = torch.tensor([1.5, 1.0])  # Higher weight for AI class
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_acc = 0
    patience = 0
    max_patience = 5
    
    # Training loop
    for epoch in range(20):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_ai_correct = 0
        train_real_correct = 0
        train_ai_total = 0
        train_real_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Track per-class accuracy
            for i, label in enumerate(labels):
                if label == 0:  # AI
                    train_ai_total += 1
                    if predicted[i] == label:
                        train_ai_correct += 1
                else:  # Real
                    train_real_total += 1
                    if predicted[i] == label:
                        train_real_correct += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_ai_correct = 0
        val_real_correct = 0
        val_ai_total = 0
        val_real_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Track per-class accuracy
                for i, label in enumerate(labels):
                    if label == 0:  # AI
                        val_ai_total += 1
                        if predicted[i] == label:
                            val_ai_correct += 1
                    else:  # Real
                        val_real_total += 1
                        if predicted[i] == label:
                            val_real_correct += 1
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        train_ai_acc = 100 * train_ai_correct / train_ai_total if train_ai_total > 0 else 0
        train_real_acc = 100 * train_real_correct / train_real_total if train_real_total > 0 else 0
        val_ai_acc = 100 * val_ai_correct / val_ai_total if val_ai_total > 0 else 0
        val_real_acc = 100 * val_real_correct / val_real_total if val_real_total > 0 else 0
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train Acc = {train_acc:.2f}% (AI: {train_ai_acc:.2f}%, Real: {train_real_acc:.2f}%)')
        print(f'  Val Acc = {val_acc:.2f}% (AI: {val_ai_acc:.2f}%, Real: {val_real_acc:.2f}%)')
        
        # Save best model based on balanced accuracy
        balanced_val_acc = (val_ai_acc + val_real_acc) / 2
        if balanced_val_acc > best_val_acc:
            best_val_acc = balanced_val_acc
            torch.save(model.state_dict(), "face_detector.pth")
            print(f'  New best model saved with balanced validation accuracy: {balanced_val_acc:.2f}%')
            patience = 0
        else:
            patience += 1
        
        scheduler.step(balanced_val_acc)
        
        # Early stopping
        if patience >= max_patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # Save class mapping
    class_to_idx = {"ai": 0, "real": 1}
    with open("class_to_idx.json", 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    print(f"Training completed! Best balanced validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    retrain_model()
