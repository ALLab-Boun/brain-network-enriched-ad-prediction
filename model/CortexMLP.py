

from typing import Optional, List
import torch
import torch.nn as nn


def get_activation(name: str = "gelu", negative_slope: float = 0.01):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=negative_slope)
    elif name == "gelu":
        return nn.GELU()
    elif name == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class MLPBlock(nn.Module):
    """
    One plain MLP block:
        Linear -> LayerNorm -> Activation -> Dropout

    Residual connections are NOT applied inside this block.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.3,
        activation: str = "gelu",
        negative_slope: float = 0.01,
        use_layernorm: bool = True,
    ):
        super().__init__()

        layers = [nn.Linear(in_dim, out_dim)]
        if use_layernorm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(get_activation(activation, negative_slope))
        layers.append(nn.Dropout(dropout))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class MLPCorticalBranch(nn.Module):
    """
    Flexible MLP cortical branch.

    Input:
        x: [B, N, F]

    Output:
        [B, out_dim]

    Modes:
      1) width_mode="constant"
         Example with hidden_dim=128, num_layers=3:
             in_dim -> 128 -> 128 -> 128 -> out_dim

      2) width_mode="shrink"
         Example with hidden_dims=[512, 256, 128]:
             in_dim -> 512 -> 256 -> 128 -> out_dim

         Or if hidden_dim=256, num_layers=3, shrink_factor=2.0:
             in_dim -> 256 -> 128 -> 64 -> out_dim

      3) width_mode="expand"
         Example with hidden_dim=128, num_layers=3, expand_factor=2.0:
             in_dim -> 128 -> 256 -> 512 -> out_dim

    Residual behavior:
      - No residual inside intermediate MLP blocks
      - Optional single residual from flattened input directly to final output
      - Since input dim is generally different from out_dim, this uses projection
    """
    def __init__(
        self,
        num_nodes: int,
        node_in_dim: int,
        out_dim: int = 128,

        # width control
        width_mode: str = "constant",   # "constant", "shrink", or "expand"
        hidden_dim: int = 128,
        num_layers: int = 3,
        hidden_dims: Optional[List[int]] = None,
        shrink_factor: float = 2.0,
        expand_factor: float = 2.0,
        min_hidden_dim: int = 32,

        # regularization / activation
        dropout: float = 0.3,
        activation: str = "gelu",
        negative_slope: float = 0.01,
        use_layernorm: bool = True,

        # residual options
        use_residual: bool = True,
        project_residual: bool = True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.node_in_dim = node_in_dim
        self.in_dim = num_nodes * node_in_dim
        self.out_dim = out_dim
        self.use_residual = use_residual
        self.project_residual = project_residual

        block_dims = self._build_hidden_dims(
            width_mode=width_mode,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            shrink_factor=shrink_factor,
            expand_factor=expand_factor,
            min_hidden_dim=min_hidden_dim,
        )

        layers = []
        prev_dim = self.in_dim

        for hdim in block_dims:
            layers.append(
                MLPBlock(
                    in_dim=prev_dim,
                    out_dim=hdim,
                    dropout=dropout,
                    activation=activation,
                    negative_slope=negative_slope,
                    use_layernorm=use_layernorm,
                )
            )
            prev_dim = hdim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, out_dim)

        # Single final residual from input -> output
        if self.use_residual:
            if self.in_dim == self.out_dim:
                self.final_shortcut = nn.Identity()
            elif self.project_residual:
                self.final_shortcut = nn.Linear(self.in_dim, self.out_dim)
            else:
                self.final_shortcut = None
        else:
            self.final_shortcut = None

    def _build_hidden_dims(
        self,
        width_mode: str,
        hidden_dim: int,
        num_layers: int,
        hidden_dims: Optional[List[int]],
        shrink_factor: float,
        expand_factor: float,
        min_hidden_dim: int,
    ) -> List[int]:
        width_mode = width_mode.lower()

        if hidden_dims is not None:
            if len(hidden_dims) == 0:
                raise ValueError("hidden_dims was provided but is empty.")
            return hidden_dims

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        if width_mode == "constant":
            return [hidden_dim] * num_layers

        elif width_mode == "shrink":
            if shrink_factor <= 1.0:
                raise ValueError("shrink_factor must be > 1.0 for shrink mode.")

            dims = []
            current = hidden_dim
            for _ in range(num_layers):
                dims.append(max(int(round(current)), min_hidden_dim))
                current = current / shrink_factor
            return dims

        elif width_mode == "expand":
            if expand_factor <= 1.0:
                raise ValueError("expand_factor must be > 1.0 for expand mode.")

            dims = []
            current = hidden_dim
            for _ in range(num_layers):
                dims.append(max(int(round(current)), min_hidden_dim))
                current = current * expand_factor
            return dims

        else:
            raise ValueError(f"Unsupported width_mode: {width_mode}")

    def forward(self, x):
        x = x.reshape(x.size(0), -1)   # [B, N*F]
        identity = x

        x = self.backbone(x)
        x = self.head(x)

        if self.final_shortcut is not None:
            x = x + self.final_shortcut(identity)

        return x
    
