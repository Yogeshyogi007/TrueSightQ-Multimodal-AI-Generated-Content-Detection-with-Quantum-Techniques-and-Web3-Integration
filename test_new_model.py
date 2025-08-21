"""
Test the newly trained balanced model
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import random

def test_model():
    # Load the model
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )
    
    model.load_state_dict(torch.load("face_detector.pth", map_location='cpu'))
    model.eval()
    
    # Load class mapping
    with open("class_to_idx.json", 'r') as f:
        class_to_idx = json.load(f)
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Test on some sample images
    print("Testing new balanced model...")
    
    # Test AI images
    ai_dir = "dataset/ai"
    ai_files = [f for f in os.listdir(ai_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    ai_sample = random.sample(ai_files, 5)
    
    print("\nTesting AI images:")
    for img_name in ai_sample:
        img_path = os.path.join(ai_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize((224, 224))
            image_tensor = transforms.ToTensor()(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                predicted_label = idx_to_class[predicted_class]
                print(f"  {img_name}: {predicted_label} ({confidence:.3f})")
        except Exception as e:
            print(f"  {img_name}: Error - {e}")
    
    # Test real images
    real_dir = "dataset/real"
    real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    real_sample = random.sample(real_files, 5)
    
    print("\nTesting real images:")
    for img_name in real_sample:
        img_path = os.path.join(real_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize((224, 224))
            image_tensor = transforms.ToTensor()(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                predicted_label = idx_to_class[predicted_class]
                print(f"  {img_name}: {predicted_label} ({confidence:.3f})")
        except Exception as e:
            print(f"  {img_name}: Error - {e}")

if __name__ == "__main__":
    test_model()
