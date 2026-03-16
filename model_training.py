# model_training.py
import torch
import torch.nn as nn

class HousePricePredictor(nn.Module):
    """
    Multimodal Architecture strictly aligned with the saved checkpoint.
    Fuses 2560 Image features and 125 Tabular features.
    """
    def __init__(self, input_dim=None, image_dim=2560):
        super(HousePricePredictor, self).__init__()
        
        # Branch 1: Image Processing Pathway (img_net)
        # Expected shape: [512, 2560]
        self.img_net = nn.Sequential(
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        
        # Branch 2: Tabular Processing Pathway (tab_net)
        # Expected shape: [128, 125]
        self.tab_net = nn.Sequential(
            nn.Linear(125, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Branch 3: Final Regression Head (regressor)
        # Expected shapes: [256, 640] -> [128, 256] -> [1, 128]
        self.regressor = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # Splitting input x: 2560 (Images) + 125 (Tabular)
        img_features = x[:, :2560]
        tab_features = x[:, 2560:2685]
        
        img_out = self.img_net(img_features)
        tab_out = self.tab_net(tab_features)
        
        # Concatenate (Late Fusion)
        combined = torch.cat((img_out, tab_out), dim=1)
        
        return self.regressor(combined)