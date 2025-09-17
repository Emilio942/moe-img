
import pytest
import torch
from models.expert_graph import ExpertGNNLayer

def test_gnn_layer_creation():
    """
    Tests if the ExpertGNNLayer can be created.
    """
    feature_dim = 16
    num_experts = 8
    layer = ExpertGNNLayer(feature_dim, num_experts)
    assert layer is not None

def test_gnn_layer_forward_shape():
    """
    Tests the output shape of the ExpertGNNLayer forward pass.
    """
    feature_dim = 16
    num_experts = 8
    batch_size = 4

    layer = ExpertGNNLayer(feature_dim, num_experts)
    expert_features = torch.randn(batch_size, num_experts, feature_dim)
    adjacency_matrix = torch.randn(num_experts, num_experts)

    output = layer.forward(expert_features, adjacency_matrix)

    assert output.shape == (batch_size, num_experts, feature_dim)

def test_gnn_layer_grad():
    """
    Tests if the gradients flow through the GNN layer.
    """
    feature_dim = 16
    num_experts = 8
    batch_size = 4

    layer = ExpertGNNLayer(feature_dim, num_experts)
    expert_features = torch.randn(batch_size, num_experts, feature_dim, requires_grad=True)
    adjacency_matrix = torch.randn(num_experts, num_experts, requires_grad=True)

    output = layer.forward(expert_features, adjacency_matrix)
    output.sum().backward()

    assert expert_features.grad is not None
    # The adjacency matrix is not a parameter of the GNN layer, so we don't check its grad here.
