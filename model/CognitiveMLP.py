import torch
import torch.nn as nn


class MLPCogBranch(nn.Module):
    """
    Flexible MLP branch for cognitive features.

    Input:
        x_cog: [B, cog_in_dim]

    Output:
        [B, out_dim]
    """

    def __init__(
        self,
        cog_in_dim: int,
        out_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        width_mode: str = "shrink",   # "constant" | "shrink" | "expand"
        width_factor: float = 2.0,
        use_residual_to_last: bool = False,
    ):
        super().__init__()

        self.use_residual_to_last = use_residual_to_last

        # build hidden layer dimensions
        hidden_dims = []
        current = hidden_dim

        for _ in range(num_layers):
            hidden_dims.append(int(current))

            if width_mode == "shrink":
                current = current / width_factor
            elif width_mode == "expand":
                current = current * width_factor
            elif width_mode == "constant":
                pass
            else:
                raise ValueError("width_mode must be 'constant', 'shrink', or 'expand'")

        layers = []
        prev_dim = cog_in_dim

        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.LayerNorm(hdim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        self.hidden = nn.Sequential(*layers)
        self.final_linear = nn.Linear(prev_dim, out_dim)

        if use_residual_to_last:
            if cog_in_dim == out_dim:
                self.residual_proj = nn.Identity()
            else:
                self.residual_proj = nn.Linear(cog_in_dim, out_dim)

    def forward(self, x_cog):
        identity = x_cog

        x = self.hidden(x_cog)
        x = self.final_linear(x)

        if self.use_residual_to_last:
            x = x + self.residual_proj(identity)

        return x