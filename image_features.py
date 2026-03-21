import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class VisualFeatureExtractor:
    def __init__(self):
        # Load pre-trained ResNet-18
        resnet = models.resnet18(pretrained=True)
        # Remove the last classification layer to get raw features (512-dim)
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image_path):
        """Converts an image file into a 512-dimensional feature vector."""
        try:
            img = Image.open(image_path).convert('RGB')
            img_t = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                features = self.model(img_t)
            return features.flatten().numpy()
        except Exception as e:
            # Fallback to zero-vector if image fails
            return np.zeros(512)

if __name__ == "__main__":
    extractor = VisualFeatureExtractor()
    print("Vision Engine Initialized. Ready to process property photos.")