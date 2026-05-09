import torch
import torch.nn as nn


class CNNAdjacency1D_Pool(nn.Module):
    def __init__(
        self,
        num_nodes,
        out_dim=128,

        # architecture
        conv_channels=(32, 256, 2048),
        kernel_sizes=(7, 5, 3),
        strides=(2, 2, 1),
        # paddings=(3, 2, 1),

        # pooling inside conv blocks
        pool_types=("max", "max", "avg"),
        pool_kernel_sizes=(4, 4, 4),

        # final readout
        readout="flatten",   # "flatten", "gap", "gmp", "gap_gmp"

        # activation
        negative_slope=0.01,

        # regularization
        dropout=0.4,

        # normalization
        norm_type=None,   # None, "batch", "group"
        group_norm_groups=8,
    ):
        super().__init__()

        input_len = num_nodes * num_nodes
        c1, c2, c3 = conv_channels
        k1, k2, k3 = kernel_sizes
        s1, s2, s3 = strides
        p1, p2, p3 = (k1 // 2, k2 // 2, k3 // 2)

        layers = []

        def add_norm(num_channels):
            if norm_type is None or str(norm_type).lower() == "none":
                return []
            elif norm_type == "batch":
                return [nn.BatchNorm1d(num_channels)]
            elif norm_type == "group":
                return [nn.GroupNorm(num_groups=group_norm_groups, num_channels=num_channels)]
            else:
                raise ValueError(f"Unsupported norm_type: {norm_type}")

        def add_pool(pool_type, kernel_size):
            if pool_type == "max":
                return nn.MaxPool1d(kernel_size)
            elif pool_type == "avg":
                return nn.AvgPool1d(kernel_size)
            elif pool_type is None or str(pool_type).lower() == "none":
                return nn.Identity()
            else:
                raise ValueError(f"Unsupported pool_type: {pool_type}")

        # block 1
        layers += [
            nn.Conv1d(1, c1, k1, stride=s1, padding=p1),
            *add_norm(c1),
            nn.LeakyReLU(negative_slope=negative_slope),
            add_pool(pool_types[0], pool_kernel_sizes[0]),
        ]

        # block 2
        layers += [
            nn.Conv1d(c1, c2, k2, stride=s2, padding=p2),
            *add_norm(c2),
            nn.LeakyReLU(negative_slope=negative_slope),
            add_pool(pool_types[1], pool_kernel_sizes[1]),
        ]

        # block 3
        layers += [
            nn.Conv1d(c2, c3, k3, stride=s3, padding=p3),
            *add_norm(c3),
            nn.LeakyReLU(negative_slope=negative_slope),
            add_pool(pool_types[2], pool_kernel_sizes[2]),
        ]

        self.conv = nn.Sequential(*layers)
        self.readout = readout

        # final pooling/readout
        if readout == "flatten":
            with torch.no_grad():
                dummy = torch.zeros(1, 1, input_len)
                conv_out = self.conv(dummy)
                feat_dim = conv_out.flatten(1).shape[1]
            self.global_pool = None

        elif readout == "gap":
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            feat_dim = c3

        elif readout == "gmp":
            self.global_pool = nn.AdaptiveMaxPool1d(1)
            feat_dim = c3

        elif readout == "gap_gmp":
            self.global_pool_avg = nn.AdaptiveAvgPool1d(1)
            self.global_pool_max = nn.AdaptiveMaxPool1d(1)
            feat_dim = 2 * c3

        else:
            raise ValueError(f"Unsupported readout: {readout}")

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, out_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            b, n, _ = x.shape
            x = x.reshape(b, 1, n * n)

        x = self.conv(x)   # [B, C, L]

        if self.readout == "flatten":
            x = x.flatten(1)

        elif self.readout in ["gap", "gmp"]:
            x = self.global_pool(x).squeeze(-1)   # [B, C]

        elif self.readout == "gap_gmp":
            x_avg = self.global_pool_avg(x).squeeze(-1)   # [B, C]
            x_max = self.global_pool_max(x).squeeze(-1)   # [B, C]
            x = torch.cat([x_avg, x_max], dim=1)          # [B, 2C]

        return self.fc(x)

