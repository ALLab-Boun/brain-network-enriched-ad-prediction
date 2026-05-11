import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, GINConv, GraphNorm


class GNNBranch_with_1dcnn_flattened(nn.Module):
    """
    Flexible GNN branch with:
      - optional 1-layer pre-MLP before GNN
      - arbitrary number of GNN layers
      - layer connectivity modes: stack / skipsum / skipcat
      - 1D CNN over flattened node embeddings
      - optional concatenation of flattened raw node features into CNN input
      - optional raw-feature skip to final output

    Connectivity modes:
      - stack:
            no skip connection
      - skipsum:
            current = current + projected(previous)
      - skipcat:
            current = projection([current, previous])

    Notes:
      - assumes fixed num_nodes=68 for flattening/CNN length inference
      - all GNN hidden representations are kept at hidden_dim
    """

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 128,
        dropout: float = 0.5,
        negative_slope: float = 0.01,

        num_gnn_layers: int = 2,
        norm_type: str = "layernorm",   # "layernorm" | "graphnorm" | "none"
        gnn_layer: str = "gcn",         # "gcn" | "sage" | "gatv2" | "gin"
        gat_heads: int = 4,
        gin_mlp_hidden: int | None = None,

        # optional pre-MLP
        use_pre_mlp: bool = True,
        pre_mlp_dim: int | None = None,
        pre_mlp_use_norm: bool = True,
        pre_mlp_use_activation: bool = True,
        pre_mlp_use_dropout: bool = True,

        # connectivity
        layer_connectivity: str = "skipcat",   # "stack" | "skipsum" | "skipcat"

        # CNN input / output skip
        cnn_input_add_flattened_node_features: bool = True,
        add_output_skip: bool = True,

        # fixed graph size used by the branch
        num_nodes: int = 68,
    ):
        super().__init__()

        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be >= 1")

        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.num_gnn_layers = num_gnn_layers
        self.num_nodes = num_nodes

        self.norm_type = norm_type.lower()
        self.gnn_layer = gnn_layer.lower()
        self.layer_connectivity = layer_connectivity.lower()

        self.use_pre_mlp = use_pre_mlp
        self.pre_mlp_use_norm = pre_mlp_use_norm
        self.pre_mlp_use_activation = pre_mlp_use_activation
        self.pre_mlp_use_dropout = pre_mlp_use_dropout

        self.cnn_input_add_flattened_node_features = cnn_input_add_flattened_node_features
        self.add_output_skip = add_output_skip

        if self.layer_connectivity not in {"stack", "skipsum", "skipcat"}:
            raise ValueError(
                "layer_connectivity must be one of: 'stack', 'skipsum', 'skipcat'"
            )

        if gin_mlp_hidden is None:
            gin_mlp_hidden = hidden_dim
        self.gin_mlp_hidden = gin_mlp_hidden

        if pre_mlp_dim is None:
            pre_mlp_dim = hidden_dim
        self.pre_mlp_dim = pre_mlp_dim

        # -------------------------------------------------
        # optional pre-MLP
        # -------------------------------------------------
        if self.use_pre_mlp:
            pre_mlp_layers = [nn.Linear(node_in_dim, self.pre_mlp_dim)]

            if self.pre_mlp_use_norm:
                pre_mlp_layers.append(nn.LayerNorm(self.pre_mlp_dim))

            if self.pre_mlp_use_activation:
                pre_mlp_layers.append(nn.LeakyReLU(self.negative_slope))

            if self.pre_mlp_use_dropout:
                pre_mlp_layers.append(nn.Dropout(self.dropout))

            self.pre_mlp = nn.Sequential(*pre_mlp_layers)
            first_layer_input_dim = self.pre_mlp_dim
        else:
            self.pre_mlp = nn.Identity()
            first_layer_input_dim = node_in_dim

        # -------------------------------------------------
        # GNN layers
        # -------------------------------------------------
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.connectivity_projs = nn.ModuleList()

        prev_dim = first_layer_input_dim
        for layer_idx in range(self.num_gnn_layers):
            # conv always outputs hidden_dim
            conv = self._make_conv(
                in_dim=prev_dim,
                out_dim=hidden_dim,
                gnn_layer=self.gnn_layer,
                gat_heads=gat_heads,
                gin_mlp_hidden=self.gin_mlp_hidden,
            )
            self.convs.append(conv)
            self.norms.append(self._make_norm(hidden_dim))

            # connectivity projection for this layer
            if self.layer_connectivity == "stack":
                proj = nn.Identity()

            elif self.layer_connectivity == "skipsum":
                # project previous representation to hidden_dim if needed
                if prev_dim == hidden_dim:
                    proj = nn.Identity()
                else:
                    proj = nn.Linear(prev_dim, hidden_dim)

            elif self.layer_connectivity == "skipcat":
                # concat [current(hidden_dim), previous(prev_dim)] -> hidden_dim
                proj = nn.Linear(hidden_dim + prev_dim, hidden_dim)

            else:
                raise ValueError(f"Unknown layer_connectivity={self.layer_connectivity}")

            self.connectivity_projs.append(proj)

            # after this layer, representation dimension is hidden_dim
            prev_dim = hidden_dim

        # -------------------------------------------------
        # CNN branch
        # -------------------------------------------------
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, stride=4, kernel_size=7, padding=3),
            nn.LeakyReLU(negative_slope),
            nn.MaxPool1d(4),

            nn.Conv1d(in_channels=128, out_channels=256, stride=3, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope),
            nn.MaxPool1d(4),

            nn.Conv1d(in_channels=256, out_channels=512, stride=2, kernel_size=3, padding=1),
            nn.AvgPool1d(4),
        )

        cnn_input_len = self.num_nodes * hidden_dim
        if self.cnn_input_add_flattened_node_features:
            cnn_input_len += self.num_nodes * node_in_dim

        with torch.no_grad():
            dummy = torch.zeros(1, 1, cnn_input_len)
            conv_out = self.cnn_branch(dummy)
            conv_out_len = conv_out.shape[-1]

        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(512 * conv_out_len, out_dim),
        )

        if self.add_output_skip:
            self.feature_proj = nn.Linear(self.num_nodes * node_in_dim, out_dim)

    # -------------------------------------------------
    # helpers
    # -------------------------------------------------
    def _make_norm(self, dim: int) -> nn.Module:
        if self.norm_type == "layernorm":
            return nn.LayerNorm(dim)
        if self.norm_type == "graphnorm":
            return GraphNorm(dim)
        if self.norm_type == "none":
            return nn.Identity()
        raise ValueError(
            f"Unknown norm_type={self.norm_type}. "
            f"Use 'layernorm', 'graphnorm', or 'none'."
        )

    def _apply_norm(
        self,
        norm: nn.Module,
        x: torch.Tensor,
        batch: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(norm, GraphNorm):
            if batch is None:
                raise ValueError("batch must be provided when using GraphNorm.")
            return norm(x, batch)
        return norm(x)

    def _make_conv(
        self,
        in_dim: int,
        out_dim: int,
        gnn_layer: str,
        gat_heads: int,
        gin_mlp_hidden: int,
    ) -> nn.Module:
        gnn_layer = gnn_layer.lower()

        if gnn_layer == "gcn":
            return GCNConv(in_dim, out_dim)

        if gnn_layer == "sage":
            return SAGEConv(in_dim, out_dim, aggr="mean")

        if gnn_layer == "gatv2":
            return GATv2Conv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=gat_heads,
                concat=False,
                dropout=self.dropout,
                negative_slope=self.negative_slope,
            )

        if gnn_layer == "gin":
            mlp = nn.Sequential(
                nn.Linear(in_dim, gin_mlp_hidden),
                nn.LeakyReLU(self.negative_slope),
                nn.Linear(gin_mlp_hidden, out_dim),
            )
            return GINConv(nn=mlp)

        raise ValueError("Unknown gnn_layer. Use one of: 'gcn', 'sage', 'gatv2', 'gin'.")

    def _conv_forward(
        self,
        conv: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(conv, GCNConv):
            return conv(x, edge_index, edge_weight=edge_weight)
        return conv(x, edge_index)

    def _apply_connectivity(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        mode: str,
        proj: nn.Module,
    ) -> torch.Tensor:
        if mode == "stack":
            return current

        if mode == "skipsum":
            return current + proj(previous)

        if mode == "skipcat":
            return proj(torch.cat([current, previous], dim=-1))

        raise ValueError(f"Unknown connectivity mode: {mode}")

    # -------------------------------------------------
    # forward
    # -------------------------------------------------
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        x:         [total_nodes, node_in_dim]
        edge_index:[2, total_edges]
        edge_attr: [total_edges] or [total_edges, 1] or None
        batch:     [total_nodes]
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        batch_size = int(batch.max().item()) + 1
        x_raw = x

        edge_weight = (
            edge_attr.squeeze(-1)
            if (edge_attr is not None and edge_attr.dim() > 1)
            else edge_attr
        )

        # optional pre-MLP
        x = self.pre_mlp(x_raw)

        # GNN stack
        for layer_idx in range(self.num_gnn_layers):
            x_prev = x

            x = self._conv_forward(
                self.convs[layer_idx],
                x,
                edge_index,
                edge_weight=edge_weight,
            )

            x = self._apply_connectivity(
                current=x,
                previous=x_prev,
                mode=self.layer_connectivity,
                proj=self.connectivity_projs[layer_idx],
            )

            x = self._apply_norm(self.norms[layer_idx], x, batch)
            x = F.leaky_relu(x, negative_slope=self.negative_slope)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_gnn = x

        # flatten for CNN
        x_gnn_flat = x_gnn.view(batch_size, -1)
        x_raw_flat = x_raw.view(batch_size, -1)

        if self.cnn_input_add_flattened_node_features:
            x_concat = torch.cat([x_gnn_flat, x_raw_flat], dim=-1)
        else:
            x_concat = x_gnn_flat

        x_concat = x_concat.unsqueeze(1)   # [B, 1, length]
        x_cnn = self.cnn_branch(x_concat)
        x_out = self.fc_out(x_cnn)

        if self.add_output_skip:
            x_skip = self.feature_proj(x_raw_flat)
            x_out = x_out + x_skip

        return x_out
    


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATv2Conv,
    GINConv,
    GraphNorm,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)


class GNNBranch_with_pooling(nn.Module):
    """
    Flexible GNN branch with:
      - optional 1-layer pre-MLP before GNN
      - arbitrary number of GNN layers
      - layer connectivity modes: stack / skipsum / skipcat
      - graph-level readout via pooling instead of 1D CNN
      - optional raw-feature skip to final output

    Connectivity modes:
      - stack:
            no skip connection
      - skipsum:
            current = current + projected(previous)
      - skipcat:
            current = projection([current, previous])

    Pooling modes:
      - "mean"
      - "max"
      - "sum"
      - "mean_max"   -> concat(mean, max)

    Notes:
      - unlike the original version, this does NOT flatten node embeddings
        for CNN processing
      - this version does NOT include
        cnn_input_add_flattened_node_features
    """

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 128,
        dropout: float = 0.5,
        negative_slope: float = 0.01,

        num_gnn_layers: int = 2,
        norm_type: str = "layernorm",   # "layernorm" | "graphnorm" | "none"
        gnn_layer: str = "gcn",         # "gcn" | "sage" | "gatv2" | "gin"
        gat_heads: int = 4,
        gin_mlp_hidden: int | None = None,

        # optional pre-MLP
        use_pre_mlp: bool = True,
        pre_mlp_dim: int | None = None,
        pre_mlp_use_norm: bool = True,
        pre_mlp_use_activation: bool = True,
        pre_mlp_use_dropout: bool = True,

        # connectivity
        layer_connectivity: str = "skipcat",   # "stack" | "skipsum" | "skipcat"

        # graph pooling readout
        graph_pool: str = "mean_max",          # "mean" | "max" | "sum" | "mean_max"
        readout_hidden_dim: int | None = None,

        # optional output skip
        add_output_skip: bool = True,

        # kept only if you still want fixed-size raw-feature skip
        num_nodes: int = 68,
    ):
        super().__init__()

        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be >= 1")

        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.num_gnn_layers = num_gnn_layers
        self.num_nodes = num_nodes

        self.norm_type = norm_type.lower()
        self.gnn_layer = gnn_layer.lower()
        self.layer_connectivity = layer_connectivity.lower()
        self.graph_pool = graph_pool.lower()

        self.use_pre_mlp = use_pre_mlp
        self.pre_mlp_use_norm = pre_mlp_use_norm
        self.pre_mlp_use_activation = pre_mlp_use_activation
        self.pre_mlp_use_dropout = pre_mlp_use_dropout

        self.add_output_skip = add_output_skip

        if self.layer_connectivity not in {"stack", "skipsum", "skipcat"}:
            raise ValueError(
                "layer_connectivity must be one of: 'stack', 'skipsum', 'skipcat'"
            )

        if self.graph_pool not in {"mean", "max", "sum", "mean_max"}:
            raise ValueError(
                "graph_pool must be one of: 'mean', 'max', 'sum', 'mean_max'"
            )

        if gin_mlp_hidden is None:
            gin_mlp_hidden = hidden_dim
        self.gin_mlp_hidden = gin_mlp_hidden

        if pre_mlp_dim is None:
            pre_mlp_dim = hidden_dim
        self.pre_mlp_dim = pre_mlp_dim

        if readout_hidden_dim is None:
            readout_hidden_dim = hidden_dim
        self.readout_hidden_dim = readout_hidden_dim

        # -------------------------------------------------
        # optional pre-MLP
        # -------------------------------------------------
        if self.use_pre_mlp:
            pre_mlp_layers = [nn.Linear(node_in_dim, self.pre_mlp_dim)]

            if self.pre_mlp_use_norm:
                pre_mlp_layers.append(nn.LayerNorm(self.pre_mlp_dim))

            if self.pre_mlp_use_activation:
                pre_mlp_layers.append(nn.LeakyReLU(self.negative_slope))

            if self.pre_mlp_use_dropout:
                pre_mlp_layers.append(nn.Dropout(self.dropout))

            self.pre_mlp = nn.Sequential(*pre_mlp_layers)
            first_layer_input_dim = self.pre_mlp_dim
        else:
            self.pre_mlp = nn.Identity()
            first_layer_input_dim = node_in_dim

        # -------------------------------------------------
        # GNN layers
        # -------------------------------------------------
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.connectivity_projs = nn.ModuleList()

        prev_dim = first_layer_input_dim
        for _ in range(self.num_gnn_layers):
            conv = self._make_conv(
                in_dim=prev_dim,
                out_dim=hidden_dim,
                gnn_layer=self.gnn_layer,
                gat_heads=gat_heads,
                gin_mlp_hidden=self.gin_mlp_hidden,
            )
            self.convs.append(conv)
            self.norms.append(self._make_norm(hidden_dim))

            if self.layer_connectivity == "stack":
                proj = nn.Identity()

            elif self.layer_connectivity == "skipsum":
                if prev_dim == hidden_dim:
                    proj = nn.Identity()
                else:
                    proj = nn.Linear(prev_dim, hidden_dim)

            elif self.layer_connectivity == "skipcat":
                proj = nn.Linear(hidden_dim + prev_dim, hidden_dim)

            self.connectivity_projs.append(proj)
            prev_dim = hidden_dim

        # -------------------------------------------------
        # pooled graph readout
        # -------------------------------------------------
        if self.graph_pool == "mean_max":
            pooled_dim = 2 * hidden_dim
        else:
            pooled_dim = hidden_dim

        self.fc_out = nn.Sequential(
            nn.Linear(pooled_dim, self.readout_hidden_dim),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(self.dropout),
            nn.Linear(self.readout_hidden_dim, out_dim),
        )

        # optional raw-input skip, projected from flattened raw node features
        if self.add_output_skip:
            self.feature_proj = nn.Linear(self.num_nodes * node_in_dim, out_dim)

    # -------------------------------------------------
    # helpers
    # -------------------------------------------------
    def _make_norm(self, dim: int) -> nn.Module:
        if self.norm_type == "layernorm":
            return nn.LayerNorm(dim)
        if self.norm_type == "graphnorm":
            return GraphNorm(dim)
        if self.norm_type == "none":
            return nn.Identity()
        raise ValueError(
            f"Unknown norm_type={self.norm_type}. "
            f"Use 'layernorm', 'graphnorm', or 'none'."
        )

    def _apply_norm(
        self,
        norm: nn.Module,
        x: torch.Tensor,
        batch: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(norm, GraphNorm):
            if batch is None:
                raise ValueError("batch must be provided when using GraphNorm.")
            return norm(x, batch)
        return norm(x)

    def _make_conv(
        self,
        in_dim: int,
        out_dim: int,
        gnn_layer: str,
        gat_heads: int,
        gin_mlp_hidden: int,
    ) -> nn.Module:
        gnn_layer = gnn_layer.lower()

        if gnn_layer == "gcn":
            return GCNConv(in_dim, out_dim)

        if gnn_layer == "sage":
            return SAGEConv(in_dim, out_dim, aggr="mean")

        if gnn_layer == "gatv2":
            return GATv2Conv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=gat_heads,
                concat=False,
                dropout=self.dropout,
                negative_slope=self.negative_slope,
            )

        if gnn_layer == "gin":
            mlp = nn.Sequential(
                nn.Linear(in_dim, gin_mlp_hidden),
                nn.LeakyReLU(self.negative_slope),
                nn.Linear(gin_mlp_hidden, out_dim),
            )
            return GINConv(nn=mlp)

        raise ValueError("Unknown gnn_layer. Use one of: 'gcn', 'sage', 'gatv2', 'gin'.")

    def _conv_forward(
        self,
        conv: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(conv, GCNConv):
            return conv(x, edge_index, edge_weight=edge_weight)
        return conv(x, edge_index)

    def _apply_connectivity(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        mode: str,
        proj: nn.Module,
    ) -> torch.Tensor:
        if mode == "stack":
            return current

        if mode == "skipsum":
            return current + proj(previous)

        if mode == "skipcat":
            return proj(torch.cat([current, previous], dim=-1))

        raise ValueError(f"Unknown connectivity mode: {mode}")

    def _pool_graph_embeddings(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.graph_pool == "mean":
            return global_mean_pool(x, batch)

        if self.graph_pool == "max":
            return global_max_pool(x, batch)

        if self.graph_pool == "sum":
            return global_add_pool(x, batch)

        if self.graph_pool == "mean_max":
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            return torch.cat([x_mean, x_max], dim=-1)

        raise ValueError(f"Unknown graph_pool mode: {self.graph_pool}")

    # -------------------------------------------------
    # forward
    # -------------------------------------------------
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        x:         [total_nodes, node_in_dim]
        edge_index:[2, total_edges]
        edge_attr: [total_edges] or [total_edges, 1] or None
        batch:     [total_nodes]
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        batch_size = int(batch.max().item()) + 1
        x_raw = x

        edge_weight = (
            edge_attr.squeeze(-1)
            if (edge_attr is not None and edge_attr.dim() > 1)
            else edge_attr
        )

        # optional pre-MLP
        x = self.pre_mlp(x_raw)

        # GNN stack
        for layer_idx in range(self.num_gnn_layers):
            x_prev = x

            x = self._conv_forward(
                self.convs[layer_idx],
                x,
                edge_index,
                edge_weight=edge_weight,
            )

            x = self._apply_connectivity(
                current=x,
                previous=x_prev,
                mode=self.layer_connectivity,
                proj=self.connectivity_projs[layer_idx],
            )

            x = self._apply_norm(self.norms[layer_idx], x, batch)
            x = F.leaky_relu(x, negative_slope=self.negative_slope)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # graph-level pooled readout
        x_graph = self._pool_graph_embeddings(x, batch)
        x_out = self.fc_out(x_graph)

        # optional raw-input skip
        if self.add_output_skip:
            x_raw_flat = x_raw.view(batch_size, self.num_nodes * self.node_in_dim)
            x_skip = self.feature_proj(x_raw_flat)
            x_out = x_out + x_skip

        return x_out