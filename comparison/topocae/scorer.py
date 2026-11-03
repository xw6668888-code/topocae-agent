"""Frozen complete-object scorer architecture."""
from torch import nn

class ObjectScorer(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_count, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)
