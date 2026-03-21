import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
# Importing the synchronized function name
from data_processing import get_processed_data

class MultimodalHousePredictor(nn.Module):
    def __init__(self, struct_dim=7, geo_dim=4):
        super(MultimodalHousePredictor, self).__init__()
        self.structural_branch = nn.Sequential(nn.Linear(struct_dim, 32), nn.ReLU())
        self.geospatial_branch = nn.Sequential(nn.Linear(geo_dim, 16), nn.ReLU())
        self.final_head = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, struct_x, geo_x):
        x1 = self.structural_branch(struct_x)
        x2 = self.geospatial_branch(geo_x)
        combined = torch.cat((x1, x2), dim=1)
        return self.final_head(combined)

def run_training():
    # Load synchronized data
    X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, feats = get_processed_data()

    # Phase 3: Baseline
    rf = RandomForestRegressor(n_estimators=50).fit(X_train_s, y_train)
    print(f"Baseline R2 Score: {rf.score(X_val_s, y_val):.4f}")

    # Phase 5: Train Multimodal DL
    model = MultimodalHousePredictor(struct_dim=7, geo_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print("--- Phase 5: Training Multimodal Model ---")
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        # Split inputs for branches (0-6: Structural, 7-10: Geospatial)
        s_t = torch.tensor(X_train_s[:, :7], dtype=torch.float32)
        g_t = torch.tensor(X_train_s[:, 7:], dtype=torch.float32)
        y_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        
        outputs = model(s_t, g_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()

    # Phase 8: Package
    torch.save(model.state_dict(), "house_price_model.pth")
    print("Model Packaged as house_price_model.pth")

if __name__ == "__main__":
    run_training()