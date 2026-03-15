# imports
import torch
import torch.nn as nn

# CNN model for flattened adjacency matrices with pooling
class CNNAdjacency1D_Pool(nn.Module):
    def __init__(self, num_nodes, num_classes=2, dropout=0.4, out_dim=128):
        super().__init__()
        input_len = num_nodes * num_nodes
        self.dropout = dropout
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3),
            # nn.GroupNorm(num_groups=8, num_channels=32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(4),       

            nn.Conv1d(32, 256, 5, stride=2, padding=2),
            # nn.GroupNorm(num_groups=8, num_channels=256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(4),          

            nn.Conv1d(256, 2048, 3, stride=1, padding=1),
            nn.AvgPool1d(4)   
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)  # (batch, channels, length)
            conv_out = self.conv(dummy)
            conv_out_len = conv_out.shape[-1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(2048 * conv_out_len, out_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            B, N, _ = x.shape
            x = x.reshape(B, 1, N * N)  # -> [B, 1, N*N]        x = self.conv(x)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)
