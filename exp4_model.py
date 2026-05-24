import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GNN import GNNBranch_with_1dcnn_flattened, GNNBranch_with_pooling
from model.CortexMLP import MLPCorticalBranch
from model.AdjacencyCNN import CNNAdjacency1D_Pool
from model.Transformer import TransformerBranch_with_1dcnn_flattened
from model.CognitiveMLP import MLPCogBranch


class FusionModel(nn.Module):
    """
    Fusion model with optional branches:
      - Cortex MLP branch
      - Cortex GNN branch
      - Cortex Transformer branch 
      - Adjacency CNN branch
      - Adjacency GNN branch
      - Adjacency Transformer branch
      - Cognitive MLP branch
    """
   
    def __init__(
        self,
        num_nodes: int,
        node_in_dim: int,
        num_classes: int = 2,
        dropout: float = 0.5,


        include_cortex_gnn: bool = True,
        include_adjacency_gnn: bool = False,   
        include_adjacency_cnn: bool = False,
        include_cortex_mlp: bool = True,
        include_cortex_transformer: bool = False,
        include_adjacency_transformer: bool = False,
        include_cog_mlp: bool = False,
        include_linear_x_logits: bool = False,
        
        separate_adj_features_instead_of_concat: bool = False,

        cortex_gnn_kwargs: dict | None = None,
        adjacency_gnn_kwargs: dict | None = None,
        cortex_mlp_kwargs: dict | None = None,
        cog_mlp_kwargs: dict | None = None,
        adjacency_cnn_kwargs: dict | None = None,
        cortex_transformer_kwargs: dict | None = None,
        adjacency_transformer_kwargs: dict | None = None,
    ):
        super().__init__()

        self.num_nodes = num_nodes
     
        self.include_cortex_gnn = include_cortex_gnn
        self.include_adjacency_gnn = include_adjacency_gnn
        self.include_adjacency_cnn = include_adjacency_cnn
        self.include_cortex_mlp = include_cortex_mlp
        self.include_cortex_transformer = include_cortex_transformer
        self.include_adjacency_transformer = include_adjacency_transformer        
        self.include_cog_mlp = include_cog_mlp
        self.include_linear_x_logits = include_linear_x_logits
        self.separate_adj_features_instead_of_concat = separate_adj_features_instead_of_concat

        self.dropout = dropout

        cortex_mlp_kwargs = cortex_mlp_kwargs or {}
        cortex_gnn_kwargs = cortex_gnn_kwargs or {}
        cortex_transformer_kwargs = cortex_transformer_kwargs or {}
        adjacency_cnn_kwargs = adjacency_cnn_kwargs or {}
        adjacency_gnn_kwargs = adjacency_gnn_kwargs or {}
        adjacency_transformer_kwargs = adjacency_transformer_kwargs or {}
        cog_mlp_kwargs = cog_mlp_kwargs or {}

        cortex_mlp_cfg = cortex_mlp_kwargs
        cortex_gnn_cfg = cortex_gnn_kwargs
        cortex_transformer_cfg = cortex_transformer_kwargs
        adjacency_cnn_cfg = adjacency_cnn_kwargs
        adjacency_gnn_cfg = adjacency_gnn_kwargs
        adjacency_transformer_cfg = adjacency_transformer_kwargs
        cog_mlp_cfg = cog_mlp_kwargs

        self.cortex_transformer_cfg = cortex_transformer_cfg
        self.adjacency_transformer_cfg = adjacency_transformer_cfg
        self.cortex_gnn_cfg = cortex_gnn_cfg
        self.adjacency_gnn_cfg = adjacency_gnn_cfg
        self.cortex_mlp_cfg = cortex_mlp_cfg
        self.cog_mlp_cfg = cog_mlp_cfg
        self.adjacency_cnn_cfg = adjacency_cnn_cfg

        # ------------------------------------------------------------------
        # Branch construction
        # ------------------------------------------------------------------

        if include_cortex_gnn:
            self.cortex_gnn = self._make_gnn_branch(
                node_in_dim=node_in_dim,
                gnn_cfg=cortex_gnn_cfg,
            )

        if include_adjacency_gnn:
            self.adjacency_gnn = self._make_gnn_branch(
                node_in_dim=num_nodes,
                gnn_cfg=adjacency_gnn_cfg,
            )

        if include_adjacency_cnn:
            self.cnn = CNNAdjacency1D_Pool(
                num_nodes,
                out_dim=128,
                dropout=adjacency_cnn_cfg["dropout"],
                conv_channels=adjacency_cnn_cfg["conv_channels"],
                kernel_sizes=adjacency_cnn_cfg["kernel_sizes"],
                strides=adjacency_cnn_cfg["strides"],
                pool_types=adjacency_cnn_cfg["pool_types"],
                pool_kernel_sizes=adjacency_cnn_cfg["pool_kernel_sizes"],
                negative_slope=adjacency_cnn_cfg["negative_slope"],
                norm_type=adjacency_cnn_cfg["norm_type"],
                group_norm_groups=adjacency_cnn_cfg["group_norm_groups"],
                readout=adjacency_cnn_cfg["readout"],
            )

        if include_cortex_mlp:
            self.mlp = MLPCorticalBranch(
                num_nodes,
                node_in_dim,
                hidden_dim=cortex_mlp_cfg["hidden_dim"],
                out_dim=128,
                dropout=cortex_mlp_cfg["dropout"],
                activation=cortex_mlp_cfg["activation"],
                use_layernorm=cortex_mlp_cfg["use_layernorm"],
                num_layers=cortex_mlp_cfg["num_layers"],
                hidden_dims=cortex_mlp_cfg["hidden_dims"],
                use_residual=cortex_mlp_cfg["use_residual"],
                width_mode=cortex_mlp_cfg["width_mode"],
            )

        if include_cortex_transformer:
            self.cortex_transformer = TransformerBranch_with_1dcnn_flattened(
                num_nodes,
                node_in_dim,
                hidden_dim=cortex_transformer_cfg["hidden_dim"],
                out_dim=128,
                num_heads=cortex_transformer_cfg["num_heads"],
                num_layers=cortex_transformer_cfg["num_layers"],
                dropout=cortex_transformer_cfg["dropout"],
                pos_encoding_type=cortex_transformer_cfg["pos_encoding_type"],
                lpe_dim=cortex_transformer_cfg["lpe_dim"],
                cnn_input_add_flattened_node_features=cortex_transformer_cfg[
                    "cnn_input_add_flattened_node_features"
                ],
                add_output_skip=cortex_transformer_cfg["add_output_skip"],
            )

        if include_adjacency_transformer:
            self.adjacency_transformer = TransformerBranch_with_1dcnn_flattened(
                num_nodes,
                num_nodes,
                hidden_dim=adjacency_transformer_cfg["hidden_dim"],
                out_dim=128,
                num_heads=adjacency_transformer_cfg["num_heads"],
                num_layers=adjacency_transformer_cfg["num_layers"],
                dropout=adjacency_transformer_cfg["dropout"],
                pos_encoding_type=adjacency_transformer_cfg["pos_encoding_type"],
                lpe_dim=adjacency_transformer_cfg["lpe_dim"],
                cnn_input_add_flattened_node_features=adjacency_transformer_cfg[
                    "cnn_input_add_flattened_node_features"
                ],
                add_output_skip=adjacency_transformer_cfg["add_output_skip"],
            )
        if include_cog_mlp:
            self.cog_branch = MLPCogBranch(
                cog_in_dim=cog_mlp_cfg["cog_in_dim"],
                hidden_dim=cog_mlp_cfg["hidden_dim"],
                out_dim=128,
                dropout=cog_mlp_cfg["dropout"],
                num_layers=cog_mlp_cfg["num_layers"],
                width_mode=cog_mlp_cfg["width_mode"],
                use_residual_to_last=cog_mlp_cfg["use_residual_to_last"],
            )

        if include_linear_x_logits:
            self.linear_x_logits = nn.Linear(
                num_nodes * node_in_dim,
                num_classes,
            )
        # ------------------------------------------------------------------
        # Final classifier
        # ------------------------------------------------------------------

        concat_dim = 0

        if include_cortex_gnn:
            concat_dim += 128

        if include_adjacency_gnn:
            concat_dim += 128

        if include_adjacency_cnn:
            concat_dim += 128

        if include_cortex_mlp:
            concat_dim += 128

        if include_cortex_transformer:
            concat_dim += 128

        if include_adjacency_transformer:
            concat_dim += 128

        if include_cog_mlp:
            concat_dim += 128

        if concat_dim == 0 and not include_linear_x_logits:
            raise ValueError(
                "At least one branch must be enabled. "
                "All include_* flags are False."
            )
        # if concat_dim == 0:
        #     raise ValueError(
        #         "At least one branch must be enabled. "
        #         "All include_* flags are False."
        #     )

        self.concat_dim = concat_dim

        if concat_dim > 0:
            self.classifier = nn.Sequential(
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(concat_dim, num_classes),
            )
        else:
            self.classifier = None

    def encode(self, data):
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        x = data.x
        lpe = getattr(data, "laplacian_pe", None)

        batch = getattr(
            data,
            "batch",
            torch.zeros(x.size(0), dtype=torch.long, device=x.device),
        )

        adj = getattr(data, "weighted_adj_matrix", None)

        zs = []

        if self.include_cortex_gnn:
            zs.append(
                self.cortex_gnn(
                    x,
                    edge_index,
                    edge_attr=edge_attr,
                    batch=batch,
                )
            )

        if self.include_adjacency_gnn:
            x_adj_row = getattr(data, "x_adj_row", None)

            if x_adj_row is None:
                raise ValueError(
                    "include_adjacency_gnn=True, but data.x_adj_row is missing. "
                    "Make sure preprocessing creates x_adj_row from weighted_adj_matrix."
                )

            zs.append(
                self.adjacency_gnn(
                    x_adj_row,
                    edge_index,
                    edge_attr=edge_attr,
                    batch=batch,
                )
            )

        if self.include_adjacency_cnn:
            if adj is None:
                raise ValueError(
                    "include_adjacency_cnn=True, but data.weighted_adj_matrix is missing."
                )
            # B = data.num_graphs
            # N = self.num_nodes

            # # Case 1: PyG batched [B*N, N]
            # if adj.dim() == 2 and adj.size(0) == B * N and adj.size(1) == N:
            #     adj = adj.view(B, N, N)
            if adj.dim() == 2:
                adj = adj.unsqueeze(0)

            # If adj is [B, 1, N, N], convert to [B, N, N].
            if adj.dim() == 4 and adj.size(1) == 1:
                adj = adj.squeeze(1)

            zs.append(self.cnn(adj))

        if self.include_cortex_mlp:
            from torch_geometric.utils import to_dense_batch

            x_dense, _ = to_dense_batch(
                x,
                batch,
                max_num_nodes=self.num_nodes,
            )

            x_flat = x_dense.view(x_dense.size(0), -1)
            zs.append(self.mlp(x_flat))

        if self.include_cortex_transformer:
            from torch_geometric.utils import to_dense_batch

            x_dense, _ = to_dense_batch(
                x,
                batch,
                max_num_nodes=self.num_nodes,
            )

            zs.append(
                self.cortex_transformer(
                    x_dense,
                    lpe=lpe,
                )
            )

        if self.include_adjacency_transformer:
            from torch_geometric.utils import to_dense_batch

            x_adj_row = getattr(data, "x_adj_row", None)

            if x_adj_row is None:
                raise ValueError(
                    "include_adjacency_transformer=True, but data.x_adj_row is missing. "
                    "Make sure preprocessing creates x_adj_row from weighted_adj_matrix."
                )

            x_adj_row_dense, _ = to_dense_batch(
                x_adj_row,
                batch,
                max_num_nodes=self.num_nodes,
            )

            zs.append(
                self.adjacency_transformer(
                    x_adj_row_dense,
                    lpe=lpe,
                )
            )
    
        if self.include_cog_mlp:
            x_cog = getattr(data, "x_cog", None)

            if x_cog is None:
                raise ValueError(
                    "include_cog_mlp=True, but data.x_cog is missing."
                )

            batch_size = data.num_graphs
            feat_dim = x_cog.numel() // batch_size
            x_cog = x_cog.view(batch_size, feat_dim)

            zs.append(self.cog_branch(x_cog))

        z_concat = torch.cat(zs, dim=-1)

        return z_concat

    def forward(self, data):
        logits = None

        if self.classifier is not None:
            z_concat = self.encode(data)
            logits = self.classifier(z_concat)

        if self.include_linear_x_logits:
            x_flat = self.get_flat_x(data)
            linear_logits = self.linear_x_logits(x_flat)

            if logits is None:
                logits = linear_logits
            else:
                logits = logits + linear_logits

        return logits
    
    def get_flat_x(self, data):
        from torch_geometric.utils import to_dense_batch

        x = data.x

        batch = getattr(
            data,
            "batch",
            torch.zeros(x.size(0), dtype=torch.long, device=x.device),
        )

        x_dense, _ = to_dense_batch(
            x,
            batch,
            max_num_nodes=self.num_nodes,
        )

        x_flat = x_dense.view(x_dense.size(0), -1)

        return x_flat

    def _make_gnn_branch(self, node_in_dim, gnn_cfg):
        if gnn_cfg["readout"] == "cnn":
            return GNNBranch_with_1dcnn_flattened(
                node_in_dim=node_in_dim,
                hidden_dim=gnn_cfg["hidden_dim"],
                out_dim=128,
                dropout=gnn_cfg["dropout"],
                num_gnn_layers=gnn_cfg["num_layers"],
                norm_type=gnn_cfg["norm_type"],
                gnn_layer=gnn_cfg["layer"],
                use_pre_mlp=gnn_cfg["use_pre_mlp"],
                cnn_input_add_flattened_node_features=gnn_cfg[
                    "cnn_input_add_flattened_node_features"
                ],
                add_output_skip=gnn_cfg["add_output_skip"],
                layer_connectivity=gnn_cfg["layer_connectivity"],
            )

        elif gnn_cfg["readout"] == "pool":
            return GNNBranch_with_pooling(
                node_in_dim=node_in_dim,
                hidden_dim=gnn_cfg["hidden_dim"],
                out_dim=128,
                dropout=gnn_cfg["dropout"],
                num_gnn_layers=gnn_cfg["num_layers"],
                norm_type=gnn_cfg["norm_type"],
                gnn_layer=gnn_cfg["layer"],
                use_pre_mlp=gnn_cfg["use_pre_mlp"],
                graph_pool=gnn_cfg["graph_pool"],
                add_output_skip=gnn_cfg["add_output_skip"],
                layer_connectivity=gnn_cfg["layer_connectivity"],
            )

        else:
            raise ValueError(
                f"Unknown gnn readout: {gnn_cfg['readout']}. "
                "Expected one of ['cnn', 'pool']."
            )
