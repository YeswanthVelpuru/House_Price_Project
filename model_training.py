# model_training.py
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from data_processing import build_training_data

MODEL_PATH = "house_price_model.pth"
SCALER_PATH = "scaler.pkl"

class HousePricePredictor(nn.Module):
    def __init__(self, total_input_dim, image_dim=2560):
        super().__init__()
        self.image_dim = image_dim
        tabular_dim = total_input_dim - image_dim
        
        # Image Feature Extractor
        self.img_net = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4)
        )
        
        # Tabular Feature Extractor
        self.tab_net = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Combined Predictor (Multimodal Fusion)
        self.regressor = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Split the input tensor into image features and tabular features
        img_feats = x[:, :self.image_dim]
        tab_feats = x[:, self.image_dim:]
        
        # Process them through their respective networks
        img_out = self.img_net(img_feats)
        tab_out = self.tab_net(tab_feats)
        
        # Concatenate and predict
        combined = torch.cat([img_out, tab_out], dim=1)
        return self.regressor(combined)

if __name__ == "__main__":
    print("Building training data (this may take a moment if images are downloading)...")
    
    # Generate the training data. limit=None uses the full dataset.
    X, y, scaler = build_training_data(use_image_cache=True, limit=None)

    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(np.float32).reshape(-1, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_np.shape[1]
    
    # Initialize the model with the split architecture
    model = HousePricePredictor(total_input_dim=input_dim, image_dim=2560).to(device)

    X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_np, dtype=torch.float32).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 15
    batch_size = 256
    n = X_tensor.size(0)
    print(f"Training on {n} samples, input dim {input_dim} on {device}")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_tensor[idx]
            yb = y_tensor[idx]

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / n
        print(f"Epoch {epoch+1}/{epochs} - Loss {avg_loss:.4f}")

    # Save model weights and the scaler
    torch.save(model.state_dict(), MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print("Saved model to", MODEL_PATH)
    print("Saved scaler to", SCALER_PATH)