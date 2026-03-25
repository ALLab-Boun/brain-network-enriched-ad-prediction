# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x shape = [B, N, d_model]
        N = x.size(1)
        return x + self.pe[:, :N, :]
    
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_nodes, hidden_dim))

    def forward(self, x):
        return x + self.pos_embed
    

from torch_geometric.transforms import AddLaplacianEigenvectorPE


class TransformerCorticalBranch_with_1dcnn_flattened(nn.Module):
    def __init__(self, num_nodes: int, node_in_dim: int, hidden_dim: int = 512,
                 out_dim: int = 128, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.5, negative_slope: float = 0.01,
                 pos_encoding_type: str = "sinusoidal",
                 lpe_dim: int = 8,
                 cnn_input_add_flattened_node_features: bool = True,
                 add_output_skip: bool = True):
        super().__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.cnn_input_add_flattened_node_features = cnn_input_add_flattened_node_features
        self.add_output_skip = add_output_skip

        #  Node + Positional Embedding 
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.pos_encoding_type = pos_encoding_type
        if pos_encoding_type == "lpe":
            self.lpe_proj = nn.Linear(lpe_dim, hidden_dim)
        elif pos_encoding_type == "learnable":
            self.pos_embed = nn.Parameter(torch.randn(1, num_nodes, hidden_dim))
        elif pos_encoding_type == "sinusoidal":
            self.pos_encoding = PositionalEncoding(hidden_dim, max_len=num_nodes)
        elif pos_encoding_type == "none":
            pass  # no positional encoding at all
        else:
            raise ValueError(f"Unknown pos_encoding_type: {pos_encoding_type}")
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 1D CNN Branch (same arch. in GNNBranch_wo_pooling_with_1dcnn_flattened) 
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=7, stride=4, padding=3),
            nn.LeakyReLU(negative_slope),
            nn.MaxPool1d(4),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=3, padding=2),
            nn.LeakyReLU(negative_slope),
            nn.MaxPool1d(4),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.AvgPool1d(4)
        )

        # Infer output size of CNN dynamically 
        cnn_input_len = self.num_nodes * hidden_dim
        if self.cnn_input_add_flattened_node_features:
            cnn_input_len += self.num_nodes * node_in_dim

        with torch.no_grad():
            dummy = torch.zeros(1, 1, cnn_input_len)  # (batch, channels, length)
            conv_out = self.cnn_branch(dummy)
            conv_out_len = conv_out.shape[-1]

        # Final fully connected projection 
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * conv_out_len, out_dim)
        )

        # Skip connection projection (optional) 
        if self.add_output_skip:
            self.feature_proj = nn.Linear(self.num_nodes * node_in_dim, out_dim)

    def forward(self, x, lpe=None):
        # x: [B, N, node_in_dim]
        batch_size = x.size(0)
        x_init = x.clone().detach()

        # Transformer encoding
        x = self.node_embed(x) # [B, N, hidden_dim]
        if self.pos_encoding_type == "lpe":
            if lpe is None:
                raise ValueError("pos_encoding_type='lpe' but lpe=None was passed to forward")

            batch_size, num_nodes = x.size(0), self.num_nodes

            if lpe.dim() == 2:
                # Two possible cases:
                # 1) [N, k]   same LPE for all graphs in batch
                # 2) [B*N, k]  concatenated LPE for a whole batch (PyG style)
                if lpe.size(0) == num_nodes:
                    # case 1: broadcast same LPE to all samples
                    lpe = lpe.unsqueeze(0).expand(batch_size, num_nodes, -1)  # [B, N, k]
                elif lpe.size(0) == batch_size * num_nodes:
                    # case 2: reshape concatenated nodes into [B, N, k]
                    lpe = lpe.view(batch_size, num_nodes, -1)                 # [B, N, k]
                else:
                    raise ValueError(
                        f"Unexpected lpe shape {lpe.shape} for batch_size={batch_size}, "
                        f"num_nodes={num_nodes}"
                    )

            elif lpe.dim() == 3:
                # assume already [B, N, k]
                if lpe.size(0) != batch_size or lpe.size(1) != num_nodes:
                    raise ValueError(
                        f"lpe shape {lpe.shape} incompatible with x shape {x.shape}"
                    )
            else:
                raise ValueError(f"Unexpected lpe dim: {lpe.dim()}")

            # Project to hidden_dim and add
            lpe_proj = self.lpe_proj(lpe)     # [B, N, hidden_dim]
            x = x + lpe_proj


        elif self.pos_encoding_type == "learnable":
            x = x + self.pos_embed

        elif self.pos_encoding_type == "sinusoidal":
            x = self.pos_encoding(x)

        elif self.pos_encoding_type == "none":
            pass  # no positional encoding applied

        else:
            raise ValueError(f"Invalid pos_encoding_type: {self.pos_encoding_type}")
        
        #         # x = self.pos_encoding(x)

        x = self.transformer(x)                              # [B, N, hidden_dim]
        # x = F.dropout(x, p=self.dropout, training=self.training)

        # Flatten node dimension
        x_flat = x.reshape(batch_size, -1)                   # [B, N * hidden_dim]
        x_init_flat = x_init.reshape(batch_size, -1)         # [B, N * node_in_dim]
        
        if self.cnn_input_add_flattened_node_features:
            x_concat = torch.cat([x_flat, x_init_flat], dim=-1)  # [B, combined_len]
        else:
            x_concat = x_flat

        # CNN branch
        x_concat = x_concat.unsqueeze(1)                     # [B, 1, L]
        x_cnn = self.cnn_branch(x_concat)                    # [B, 512, conv_out_len]
        x_cnn = x_cnn.squeeze(-1)                            # [B, 512] (after pooling)

        # Fully connected projection
        x_out = self.fc_out(x_cnn)                           # [B, out_dim]

        # Skip connection
        if self.add_output_skip:
            x_skip = self.feature_proj(x_init_flat)
            x_out = x_out + x_skip

        return x_out