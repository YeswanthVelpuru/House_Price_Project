# model_training.py
import torch
import torch.nn as nn

class HousePricePredictor(nn.Module):
    """
    Multimodal Architecture strictly aligned with the saved checkpoint.
    Matches the specific 64-unit hidden layer in the regressor head.
    """
    def __init__(self, input_dim=2685, image_dim=2560):
        super(HousePricePredictor, self).__init__()
        
        # Branch 1: Image Processing (img_net) - 2560 -> 512
        self.img_net = nn.Sequential(
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        
        # Branch 2: Tabular Processing (tab_net) - 125 -> 128
        self.tab_net = nn.Sequential(
            nn.Linear(125, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Branch 3: Regression Head (regressor)
        # Combined input: 512 + 128 = 640
        self.regressor = nn.Sequential(
            nn.Linear(640, 256),    # Layer 0
            nn.ReLU(),              # Layer 1
            nn.Dropout(0.2),        # Layer 2
            nn.Linear(256, 64),     # Layer 3: FIXED to 64 to match .pth
            nn.ReLU(),              # Layer 4
            nn.Linear(64, 1)        # Layer 5: FIXED to 64->1 to match .pth
        )
        
    def forward(self, x):
        # Input split: [0:2560] images, [2560:2685] tabular
        img_features = x[:, :2560]
        tab_features = x[:, 2560:2685]
        
        img_out = self.img_net(img_features)
        tab_out = self.tab_net(tab_features)
        
        # Concatenate features
        combined = torch.cat((img_out, tab_out), dim=1)
        
        return self.regressor(combined)