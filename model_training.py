import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from data_processing import get_processed_data

# Step 5: Define Deep Learning Architecture
class HousePricePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

def run_training():
    # Load already scaled data from Phase 2/4
    X_train, X_val, X_test, y_train, y_val, y_test, feat_names = get_processed_data()

    # Step 3: Baseline (Random Forest)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    print(f"Baseline (RF) R2 Score: {rf.score(X_val, y_val):.4f}")

    # Step 5: Train Deep Learning Model
    print("--- Phase 5: Training Deep Learning Model ---")
    model = HousePricePredictor(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(y_train.values, dtype=torch.float32).view(-1,1))
        loss.backward()
        optimizer.step()

    # Step 6: Evaluate (MSE)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")

    # Step 8: Package (ONNX & PTH)
    print("--- Phase 8: Packaging ---")
    dummy_input = torch.randn(1, X_train.shape[1])
    torch.onnx.export(model, dummy_input, "house_price_model.onnx")
    torch.save(model.state_dict(), "house_price_model.pth")
    print("Models saved successfully.")

if __name__ == "__main__":
    run_training()