import torch
from models.expert_graph import ExpertGNNLayer

def test_message_passing_shape():
    """Tests the output shape of the message passing layer."""
    feature_dim = 16
    num_experts = 8
    batch_size = 4

    layer = ExpertGNNLayer(feature_dim, num_experts)
    expert_features = torch.randn(batch_size, num_experts, feature_dim)
    adjacency_matrix = torch.randn(num_experts, num_experts)

    output = layer(expert_features, adjacency_matrix)

    assert output.shape == (batch_size, num_experts, feature_dim)

def test_message_passing_residual():
    """Tests if the residual connection is working."""
    feature_dim = 16
    num_experts = 8
    batch_size = 4

    layer = ExpertGNNLayer(feature_dim, num_experts)
    expert_features = torch.randn(batch_size, num_experts, feature_dim)
    adjacency_matrix = torch.zeros(num_experts, num_experts)  # No messages passed

    output = layer(expert_features, adjacency_matrix)

    # With a zero adjacency matrix, the output should be the normalized input
    assert torch.allclose(output, layer.norm(expert_features))

def test_message_passing_stability():
    """Tests for Inf or NaN values in the output."""
    feature_dim = 16
    num_experts = 8
    batch_size = 4

    layer = ExpertGNNLayer(feature_dim, num_experts)
    expert_features = torch.randn(batch_size, num_experts, feature_dim)
    adjacency_matrix = torch.randn(num_experts, num_experts)

    output = layer(expert_features, adjacency_matrix)

    assert not torch.isinf(output).any()
    assert not torch.isnan(output).any()
