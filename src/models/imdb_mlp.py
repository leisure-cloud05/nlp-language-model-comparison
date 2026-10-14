"""MLP classifier for frozen IMDb language-model representations."""

import torch
from torch import nn


class IMDbMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()

        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Model dimensions must be positive.")

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features):
        if features.ndim != 2:
            raise ValueError("Expected features with shape [B, D].")

        return self.network(features)