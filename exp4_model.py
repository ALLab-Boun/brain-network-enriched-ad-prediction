import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GNN import GNNBranch_wo_pooling_with_1dcnn_flattened
from model.CortexMLP import MLPCorticalBranch
from model.AdjacencyCNN import CNNAdjacency1D_Pool
from model.CortexTransformer import TransformerCorticalBranch_with_1dcnn_flattened
from model.CognitiveMLP import MLPCogBranch


# Fusion Model (Concatenation-based)
class FusionModel(nn.Module):
    def __init__(self, num_nodes: int, node_in_dim: int, num_classes: int = 2,
                 hidden_dim: int = 128, gnn_hidden_dim: int = 256, cog_hidden_dim: int = 128,
                 cortex_mlp_hidden_dim: int = 256, transformer_hidden_dim: int = 512,
                 dropout: float = 0.7, gnn_dropout: float = 0.5, adj_cnn_dropout: float = 0.5, cog_mlp_dropout: float = 0.5,
                 cort_transformer_dropout: float = 0.5, cortex_mlp_dropout: float = 0.5,

                # GNN
                gnn_use_pre_mlp=False,
                gnn_cnn_input_add_flattened_node_features = False,
                gnn_add_output_skip = False,
                gnn_layer_connectivity = "skipcat",
                gnn_layer: str = "gcn",
                gnn_num_layers: int = 2,
                gnn_norm_type   = "layernorm", 

                
                # Cortex MLP
                cortex_mlp_use_residual = False, cortex_mlp_activation = "leakyrelu",
                cortex_mlp_use_layernorm = True, cortex_mlp_num_layers = 3,
                cortex_mlp_hidden_dims = None,  # if None, will use cortex_mlp_hidden_dim and width_mode to determine hidden dims
                cortex_mlp_width_mode = "constant",  # "constant", "shrink", or "expand"

                # Cognitive MLP
                cog_mlp_num_layers = 2, cog_mlp_width_mode = "constant", cog_mlp_use_residual_to_last = False,
                
                # Adjacency CNN
                adj_cnn_conv_channels=(32, 256, 2048), adj_cnn_kernel_sizes=(7, 5, 3), 
                adj_cnn_strides=(2, 2, 1), 
                adj_cnn_pool_types=("max", "max", "avg"), adj_cnn_pool_kernel_sizes=(4, 4, 4),
                adj_cnn_negative_slope=0.01, adj_cnn_norm_type=None, adj_cnn_group_norm_groups=8,
                adj_cnn_readout="flatten",  
                
                include_gnn=True, include_cnn=False, include_mlp=True, include_transformer=False, include_cog_mlp=False,
                pos_encoding_type="sinusoidal", lpe_dim=8, cog_in_dim=6,
                 enable_modality_dropout: bool = True, p_drop_graph: float = 0.15, p_drop_cog: float = 0.35, modality_dropout_rescale: bool = True,
                  separate_adj_features_instead_of_concat: bool = False):
        super().__init__()

        # Branch configurations
        self.include_gnn = include_gnn
        self.include_cnn = include_cnn
        self.include_mlp = include_mlp
        self.include_transformer = include_transformer
        self.include_cog_mlp = include_cog_mlp
        self.num_nodes = num_nodes
        self.separate_adj_features_instead_of_concat = separate_adj_features_instead_of_concat

        # GNN configurations
        self.gnn_use_pre_mlp = gnn_use_pre_mlp
        self.gnn_cnn_input_add_flattened_node_features = gnn_cnn_input_add_flattened_node_features
        self.gnn_add_output_skip = gnn_add_output_skip
        self.gnn_layer_connectivity = gnn_layer_connectivity
        self.gnn_layer = gnn_layer
        self.gnn_num_layers = gnn_num_layers
        self.gnn_norm_type = gnn_norm_type

        # Cortex MLP
        self.cortex_mlp_use_residual = cortex_mlp_use_residual 
        self.cortex_mlp_activation = cortex_mlp_activation
        self.cortex_mlp_use_layernorm = cortex_mlp_use_layernorm 
        self.cortex_mlp_num_layers = cortex_mlp_num_layers
        self.cortex_mlp_hidden_dims = cortex_mlp_hidden_dims
        self.cortex_mlp_width_mode = cortex_mlp_width_mode

        # Cognitive MLP
        self.cog_mlp_num_layers = cog_mlp_num_layers
        self.cog_mlp_width_mode = cog_mlp_width_mode
        self.cog_mlp_use_residual_to_last = cog_mlp_use_residual_to_last

        # Adjacency CNN
        self.adj_cnn_conv_channels = adj_cnn_conv_channels
        self.adj_cnn_kernel_sizes = adj_cnn_kernel_sizes
        self.adj_cnn_strides = adj_cnn_strides
        self.adj_cnn_pool_types = adj_cnn_pool_types
        self.adj_cnn_pool_kernel_sizes = adj_cnn_pool_kernel_sizes
        self.adj_cnn_negative_slope = adj_cnn_negative_slope
        self.adj_cnn_norm_type = adj_cnn_norm_type
        self.adj_cnn_group_norm_groups = adj_cnn_group_norm_groups
        self.adj_cnn_dropout = adj_cnn_dropout
        self.adj_cnn_readout = adj_cnn_readout

        # Configurations
        self.dropout = dropout # dropout for final classifier

        # branch-specific dropouts
        self.gnn_dropout = gnn_dropout
        self.cog_mlp_dropout = cog_mlp_dropout  
        self.cort_transformer_dropout = cort_transformer_dropout
        self.cortex_mlp_dropout = cortex_mlp_dropout
        

        # TODO
        # modality dropout
        self.enable_modality_dropout = bool(enable_modality_dropout)
        self.p_drop_graph = float(p_drop_graph)
        self.p_drop_cog = float(p_drop_cog)
        self.modality_dropout_rescale = bool(modality_dropout_rescale)


        # Branches 
        if include_gnn:
            if not self.separate_adj_features_instead_of_concat:
                self.gnn = GNNBranch_wo_pooling_with_1dcnn_flattened(node_in_dim=node_in_dim, hidden_dim=gnn_hidden_dim, 
                        out_dim=128, dropout=self.gnn_dropout, num_gnn_layers=self.gnn_num_layers, norm_type="layernorm", 
                        gnn_layer=gnn_layer, use_pre_mlp=self.gnn_use_pre_mlp, cnn_input_add_flattened_node_features=self.gnn_cnn_input_add_flattened_node_features,
                        add_output_skip=self.gnn_add_output_skip, layer_connectivity=self.gnn_layer_connectivity)
            else:
                self.gnn = GNNBranch_wo_pooling_with_1dcnn_flattened(node_in_dim=num_nodes, hidden_dim=gnn_hidden_dim, 
                        out_dim=128, dropout=self.gnn_dropout, num_gnn_layers=self.gnn_num_layers, norm_type="layernorm", 
                        gnn_layer=gnn_layer, use_pre_mlp=self.gnn_use_pre_mlp, cnn_input_add_flattened_node_features=self.gnn_cnn_input_add_flattened_node_features,
                        add_output_skip=self.gnn_add_output_skip, layer_connectivity=self.gnn_layer_connectivity)
        if include_cnn:
            self.cnn = CNNAdjacency1D_Pool(num_nodes, 
                                          out_dim=128,
                                          dropout=self.adj_cnn_dropout,
                                          conv_channels=self.adj_cnn_conv_channels,
                                          kernel_sizes=self.adj_cnn_kernel_sizes,
                                          strides=self.adj_cnn_strides,
                                          pool_types=self.adj_cnn_pool_types,
                                          pool_kernel_sizes=self.adj_cnn_pool_kernel_sizes,
                                          negative_slope=self.adj_cnn_negative_slope,
                                          norm_type=self.adj_cnn_norm_type,
                                          group_norm_groups=self.adj_cnn_group_norm_groups,
                                          readout=self.adj_cnn_readout)
        if include_mlp:
            self.mlp = MLPCorticalBranch(num_nodes, node_in_dim, hidden_dim=cortex_mlp_hidden_dim, out_dim=128, 
                                         dropout=self.cortex_mlp_dropout,
                                         activation=self.cortex_mlp_activation,
                                         use_layernorm=self.cortex_mlp_use_layernorm,
                                         num_layers=self.cortex_mlp_num_layers,
                                         hidden_dims=self.cortex_mlp_hidden_dims,
                                         use_residual=self.cortex_mlp_use_residual,
                                         width_mode=self.cortex_mlp_width_mode)
        if include_transformer:
            self.transformer = TransformerCorticalBranch_with_1dcnn_flattened(num_nodes, node_in_dim, hidden_dim=transformer_hidden_dim, out_dim=128, dropout=self.cort_transformer_dropout, pos_encoding_type=pos_encoding_type,lpe_dim=lpe_dim)
        if include_cog_mlp:
            self.cog_branch = MLPCogBranch(cog_in_dim=cog_in_dim, hidden_dim=cog_hidden_dim, out_dim=128, 
                                           dropout=self.cog_mlp_dropout, num_layers=self.cog_mlp_num_layers,
                                           width_mode=self.cog_mlp_width_mode, use_residual_to_last=self.cog_mlp_use_residual_to_last)
            # if self.add_demographic_features:
            # TODO                

        # Determine total concatenated feature size 
        concat_dim = 0
        if include_gnn:
            concat_dim += 128
        if include_cnn:
            concat_dim += 128
        if include_mlp:
            concat_dim += 128
        if include_transformer:
            concat_dim += 128
        if include_cog_mlp:
            concat_dim += 128

        # Final classifier 
        self.classifier = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(concat_dim, num_classes)
        )

    def forward(self, data):
        edge_index, edge_attr = data.edge_index, getattr(data, "edge_attr", None)
        x, x_cog, lpe = data.x, getattr(data, "x_cog", None), getattr(data, "laplacian_pe", None)
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        adj = getattr(data, "weighted_adj_matrix", None)
        # print("lpe has shape", lpe.shape if lpe is not None else None)
        # print("adj has shape", adj.shape if adj is not None else None)
        # print("x_cog has shape", x_cog.shape if x_cog is not None else None)
        zs = []

        # Collect branch outputs
        if self.include_gnn:
            if not self.separate_adj_features_instead_of_concat:
                zs.append(self.gnn(x, edge_index, edge_attr=edge_attr, batch=batch))
            else:
                x_adj_row = getattr(data, "x_adj_row", None)
                zs.append(self.gnn(x_adj_row, edge_index, edge_attr=edge_attr, batch=batch))
        if self.include_cnn:
            if adj is not None and adj.dim() == 2:
                adj = adj.unsqueeze(0)
            zs.append(self.cnn(adj))
        if self.include_mlp:
            from torch_geometric.utils import to_dense_batch
            x_dense, mask = to_dense_batch(x, batch, max_num_nodes=self.num_nodes)
            x_flat = x_dense.view(x_dense.size(0), -1)
            zs.append(self.mlp(x_flat))
        if self.include_transformer:
            from torch_geometric.utils import to_dense_batch
            x_dense, mask = to_dense_batch(x, batch, max_num_nodes=self.num_nodes)

            # keep as [B, num_nodes, node_in_dim]
            zs.append(self.transformer(x_dense, lpe=lpe))
        if self.include_cog_mlp:
            batch_size = data.num_graphs  # number of graphs in the batch
            feat_dim = data.x_cog.numel() // batch_size
            x_cog = data.x_cog.view(batch_size, feat_dim)
            zs.append(self.cog_branch(x_cog))
            
        # Concatenate embeddings 
        z_concat = torch.cat(zs, dim=-1)  # [B, concat_dim]

        # Classify
        logits = self.classifier(z_concat)
        return logits


