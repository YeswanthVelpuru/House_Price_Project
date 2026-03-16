<<<<<<< HEAD
# image_features.py
import os
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision import models, transforms

# Set your Mapbox token here or in a .env file
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "YOUR_MAPBOX_TOKEN_HERE")
CACHE_DIR = "image_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Pre-trained CNN for Street-view and Satellite feature extraction
try:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    _cnn = models.efficientnet_b0(weights=weights)
    _cnn = torch.nn.Sequential(*list(_cnn.children())[:-1]) # Keep feature extractor only
    _cnn.eval()
except Exception:
    _cnn = None

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def fetch_mapbox_image(lat, lon, style="satellite-v9", zoom=16, size=512):
    """Fetch static map from Mapbox. Returns PIL.Image or None."""
    if not MAPBOX_TOKEN or MAPBOX_TOKEN == "YOUR_MAPBOX_TOKEN_HERE":
        # Return none if API key is missing (will fallback to zero-tensors)
        return None
        
    url = (f"https://api.mapbox.com/styles/v1/mapbox/{style}/static/"
           f"{lon},{lat},{zoom}/{size}x{size}?access_token={MAPBOX_TOKEN}")
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        pass
    return None

def get_image_features(lat, lon, style="satellite-v9", use_cache=True):
    """Returns 1280-d numpy vector for the requested style."""
    cache_path = os.path.join(CACHE_DIR, f"{style}_{lat:.6f}_{lon:.6f}.npy".replace(" ", "_"))
    
    if use_cache and os.path.exists(cache_path):
        return np.load(cache_path)

    img = fetch_mapbox_image(lat, lon, style)
    
    if img is not None and _cnn is not None:
        x = _transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = _cnn(x).view(-1).cpu().numpy().astype(np.float32)
    else:
        # Fallback to zeros if image download fails or no token
        feat = np.zeros(1280, dtype=np.float32)

    if use_cache:
        np.save(cache_path, feat)
        
    return feat

def calculate_greenery_index(pil_image):
    """
    Calculates a visible-spectrum greenery index (Excess Green) from an RGB image.
    Formula: ExG = 2G - R - B
    """
    if pil_image is None:
        return 0.0
        
    # Convert image to numpy array and normalize to 0-1
    img_array = np.array(pil_image).astype(np.float32) / 255.0
    
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]
    
    # Calculate Excess Green Index
    exg = (2 * G) - R - B
    
    # Take the mean across the image and scale to a 0-100 score
    green_score = np.clip(np.mean(exg) * 100, 0, 100)
    
    return float(green_score)

def get_greenery_score(lat, lon, style="satellite-v9"):
    """Fetches the image and returns the greenery score for the dashboard."""
    img = fetch_mapbox_image(lat, lon, style)
=======
# image_features.py
import os
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision import models, transforms

# Set your Mapbox token here or in a .env file
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "YOUR_MAPBOX_TOKEN_HERE")
CACHE_DIR = "image_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Pre-trained CNN for Street-view and Satellite feature extraction
try:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    _cnn = models.efficientnet_b0(weights=weights)
    _cnn = torch.nn.Sequential(*list(_cnn.children())[:-1]) # Keep feature extractor only
    _cnn.eval()
except Exception:
    _cnn = None

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def fetch_mapbox_image(lat, lon, style="satellite-v9", zoom=16, size=512):
    """Fetch static map from Mapbox. Returns PIL.Image or None."""
    if not MAPBOX_TOKEN or MAPBOX_TOKEN == "YOUR_MAPBOX_TOKEN_HERE":
        # Return none if API key is missing (will fallback to zero-tensors)
        return None
        
    url = (f"https://api.mapbox.com/styles/v1/mapbox/{style}/static/"
           f"{lon},{lat},{zoom}/{size}x{size}?access_token={MAPBOX_TOKEN}")
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        pass
    return None

def get_image_features(lat, lon, style="satellite-v9", use_cache=True):
    """Returns 1280-d numpy vector for the requested style."""
    cache_path = os.path.join(CACHE_DIR, f"{style}_{lat:.6f}_{lon:.6f}.npy".replace(" ", "_"))
    
    if use_cache and os.path.exists(cache_path):
        return np.load(cache_path)

    img = fetch_mapbox_image(lat, lon, style)
    
    if img is not None and _cnn is not None:
        x = _transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = _cnn(x).view(-1).cpu().numpy().astype(np.float32)
    else:
        # Fallback to zeros if image download fails or no token
        feat = np.zeros(1280, dtype=np.float32)

    if use_cache:
        np.save(cache_path, feat)
        
    return feat

def calculate_greenery_index(pil_image):
    """
    Calculates a visible-spectrum greenery index (Excess Green) from an RGB image.
    Formula: ExG = 2G - R - B
    """
    if pil_image is None:
        return 0.0
        
    # Convert image to numpy array and normalize to 0-1
    img_array = np.array(pil_image).astype(np.float32) / 255.0
    
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]
    
    # Calculate Excess Green Index
    exg = (2 * G) - R - B
    
    # Take the mean across the image and scale to a 0-100 score
    green_score = np.clip(np.mean(exg) * 100, 0, 100)
    
    return float(green_score)

def get_greenery_score(lat, lon, style="satellite-v9"):
    """Fetches the image and returns the greenery score for the dashboard."""
    img = fetch_mapbox_image(lat, lon, style)
>>>>>>> f1aeee6a5cef5e6f83d15321fd5985151c9c038d
    return calculate_greenery_index(img)