# imports
import torch
import torch.nn as nn

# MLP Branch for cognitive features
class MLPCogBranch(nn.Module):
    def __init__(self, cog_in_dim: int, out_dim: int = 128,
                 hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cog_in_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, out_dim),
            # nn.LayerNorm(out_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
        )

    def forward(self, x_cog):
        return self.net(x_cog)  # [B, 128]
