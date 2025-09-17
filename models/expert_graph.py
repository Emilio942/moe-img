
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertGraph(nn.Module):
    """
    Manages the cooperation of experts through a learnable graph structure.
    """
    def __init__(self, num_experts, feature_dim, aggregation_type='weighted_sum', graph_init_type='random'):
        """
        Initializes the ExpertGraph.

        Args:
            num_experts (int): The number of experts in the model (E).
            feature_dim (int): The dimension of the expert output features.
            aggregation_type (str): 'weighted_sum' or 'gnn'.
            graph_init_type (str): 'random' or 'identity'.
        """
        super().__init__()
        self.num_experts = num_experts
        self.feature_dim = feature_dim
        self.aggregation_type = aggregation_type

        # Learnable adjacency matrix representing expert interactions
        if graph_init_type == 'identity':
            self.adjacency_matrix = nn.Parameter(torch.eye(num_experts))
        else:
            self.adjacency_matrix = nn.Parameter(torch.randn(num_experts, num_experts))

        if self.aggregation_type == 'gnn':
            self.gnn_layer = ExpertGNNLayer(feature_dim, num_experts)

    def forward(self, expert_outputs, top_k_indices, top_k_weights=None, use_adjacency=True):
        """
        Performs the forward pass, aggregating outputs from the selected experts.

        Args:
            expert_outputs (torch.Tensor): The outputs from all experts (Batch, Num_Experts, Features) or (Num_Experts, Features).
            top_k_indices (torch.Tensor): The indices of the top-k experts selected by the gate (Batch, k) or (k,).
            top_k_weights (torch.Tensor, optional): Weights for the selected experts (Batch, k) or (k,). If provided, will be combined with adjacency-derived weights. Defaults to None.
            use_adjacency (bool): If False, ignore adjacency and use only top_k_weights (or uniform). Useful for warmup.

        Returns:
            torch.Tensor: The aggregated output.
        """
        if self.aggregation_type == 'gnn':
            # Update expert features with GNN layer before selection
            is_batch = expert_outputs.dim() == 3
            if not is_batch:
                expert_outputs = expert_outputs.unsqueeze(0)
            
            expert_outputs = self.gnn_layer(expert_outputs, self.adjacency_matrix)

            if not is_batch:
                expert_outputs = expert_outputs.squeeze(0)


        is_batch = expert_outputs.dim() == 3
        if not is_batch:
            expert_outputs = expert_outputs.unsqueeze(0)
            top_k_indices = top_k_indices.unsqueeze(0)

        batch_size, num_experts, feature_dim = expert_outputs.shape
        k = top_k_indices.shape[1]

        # Expand indices for gathering
        indices = top_k_indices.unsqueeze(-1).expand(batch_size, k, feature_dim)
        selected_outputs = expert_outputs.gather(1, indices)  # shape: (B, k, F)

        # Use the adjacency matrix and/or provided weights for aggregation
        aggregated_outputs = []
        for i in range(batch_size):
            # Get the sub-adjacency matrix for the selected experts
            sub_adjacency = self.adjacency_matrix[top_k_indices[i], :][:, top_k_indices[i]]  # shape: (k, k)

            # Provided team weights from gate
            if top_k_weights is not None:
                w_sel = F.softmax(top_k_weights[i], dim=0)  # (k,)
            else:
                w_sel = torch.full((k,), 1.0 / k, device=expert_outputs.device, dtype=expert_outputs.dtype)

            if use_adjacency:
                # Adjacency-derived influence. We average the influence on each expert.
                adj_influence = sub_adjacency.mean(dim=0)  # (k,)
                cooperation_gate = torch.sigmoid(adj_influence)
                
                # Modulate the original gate weights
                weights = w_sel * cooperation_gate
                
                # Re-normalize to ensure they sum to 1
                weights = weights / (weights.sum() + 1e-12)
            else:
                weights = w_sel

            # Weighted sum of expert outputs
            current_selected_outputs = selected_outputs[i]  # shape: (k, F)

            # weights need to be (k, 1) to broadcast
            weighted_output = (current_selected_outputs * weights.unsqueeze(-1)).sum(dim=0)  # shape: (F,)
            aggregated_outputs.append(weighted_output)

        aggregated_output = torch.stack(aggregated_outputs)

        if not is_batch:
            aggregated_output = aggregated_output.squeeze(0)

        return aggregated_output

    def print_graph(self):
        """
        Prints the current state of the adjacency matrix.
        """
        print("Expert Adjacency Matrix:")
        print(self.adjacency_matrix.detach().cpu().numpy())


class ExpertGNNLayer(nn.Module):
    """
    A single layer for message passing among experts.
    """
    def __init__(self, feature_dim, num_experts):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts

        # MLP for processing concatenated expert features
        self.mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # Layer normalization
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, expert_features, adjacency_matrix):
        """
        Performs one round of message passing.

        Args:
            expert_features (torch.Tensor): Features of the experts (Batch, Num_Experts, Features).
            adjacency_matrix (torch.Tensor): Learnable adjacency matrix (Num_Experts, Num_Experts).

        Returns:
            torch.Tensor: Updated expert features.
        """
        batch_size, num_experts, feature_dim = expert_features.shape

        # Expand features for pairwise concatenation
        h_i = expert_features.unsqueeze(2).expand(-1, -1, num_experts, -1)
        h_j = expert_features.unsqueeze(1).expand(-1, num_experts, -1, -1)

        # Concatenate pairs of features
        h_cat = torch.cat([h_i, h_j], dim=-1) # (B, E, E, 2*F)

        # Process with MLP
        messages = self.mlp(h_cat) # (B, E, E, F)

        # Weight messages by adjacency matrix
        adj_matrix_expanded = adjacency_matrix.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, feature_dim)
        weighted_messages = messages * adj_matrix_expanded

        # Aggregate messages
        aggregated_messages = weighted_messages.sum(dim=2) # (B, E, F)

        # Residual connection and normalization
        updated_features = self.norm(expert_features + aggregated_messages)

        return updated_features

    # No print_graph here; the adjacency matrix lives in ExpertGraph.
