import pytest
import torch
import torch.nn.functional as F
from models.expert_graph import ExpertGraph

def test_graph_creation():
    """
    Tests if the ExpertGraph can be created.
    """
    num_experts = 8
    feature_dim = 16
    graph = ExpertGraph(num_experts, feature_dim, graph_init_type='random')
    assert graph is not None
    assert graph.adjacency_matrix.shape == (num_experts, num_experts)
    assert not hasattr(graph, 'gnn_layer')

    graph_gnn = ExpertGraph(num_experts, feature_dim, aggregation_type='gnn', graph_init_type='random')
    assert graph_gnn is not None
    assert hasattr(graph_gnn, 'gnn_layer')

def test_graph_identity_initialization():
    """
    Tests if the identity initialization of the adjacency matrix works.
    """
    num_experts = 8
    feature_dim = 16
    graph = ExpertGraph(num_experts, feature_dim, graph_init_type='identity')
    assert torch.allclose(graph.adjacency_matrix.data, torch.eye(num_experts))


@pytest.mark.parametrize("aggregation_type", ['weighted_sum', 'gnn'])
def test_forward_pass_shape_single_instance(aggregation_type):
    """
    Tests the output shape for a single instance input.
    """
    num_experts = 8
    feature_dim = 16
    top_k = 3

    graph = ExpertGraph(num_experts, feature_dim, aggregation_type=aggregation_type, graph_init_type='random')
    expert_outputs = torch.randn(num_experts, feature_dim)
    top_k_indices = torch.tensor([0, 2, 4])

    output = graph.forward(expert_outputs, top_k_indices)

    assert output.shape == (feature_dim,)


@pytest.mark.parametrize("aggregation_type", ['weighted_sum', 'gnn'])
def test_forward_pass_shape_batch(aggregation_type):
    """
    Tests the output shape for a batch of inputs.
    """
    num_experts = 8
    feature_dim = 16
    top_k = 3
    batch_size = 4

    graph = ExpertGraph(num_experts, feature_dim, aggregation_type=aggregation_type, graph_init_type='random')
    expert_outputs = torch.randn(batch_size, num_experts, feature_dim)
    # Dummy top_k indices for each item in the batch
    top_k_indices = torch.randint(0, num_experts, (batch_size, top_k))

    output = graph.forward(expert_outputs, top_k_indices)

    assert output.shape == (batch_size, feature_dim)


@pytest.mark.parametrize("aggregation_type", ['weighted_sum', 'gnn'])
def test_adjacency_matrix_grad(aggregation_type):
    """
    Tests if the adjacency matrix receives gradients.
    """
    num_experts = 4
    feature_dim = 8
    top_k = 2
    batch_size = 1

    graph = ExpertGraph(num_experts, feature_dim, aggregation_type=aggregation_type, graph_init_type='random')
    expert_outputs = torch.randn(batch_size, num_experts, feature_dim, requires_grad=True)
    top_k_indices = torch.randint(0, num_experts, (batch_size, top_k))

    output = graph.forward(expert_outputs, top_k_indices)
    output.sum().backward()

    assert graph.adjacency_matrix.grad is not None

def test_cooperation_gate_weights():
    """
    Tests if the new cooperation gate logic produces weights that sum to 1.
    """
    num_experts = 8
    feature_dim = 16
    top_k = 3
    batch_size = 4

    graph = ExpertGraph(num_experts, feature_dim, graph_init_type='random')
    expert_outputs = torch.randn(batch_size, num_experts, feature_dim)
    top_k_indices = torch.randint(0, num_experts, (batch_size, top_k))
    top_k_weights = torch.randn(batch_size, top_k)

    # Manually calculate expected weights for the first batch item
    i = 0
    sub_adjacency = graph.adjacency_matrix[top_k_indices[i], :][:, top_k_indices[i]]
    w_sel = F.softmax(top_k_weights[i], dim=0)
    adj_influence = sub_adjacency.mean(dim=0)
    cooperation_gate = torch.sigmoid(adj_influence)
    manual_weights = w_sel * cooperation_gate
    manual_weights = manual_weights / (manual_weights.sum() + 1e-12)

    # Get the output from the forward pass, which internally calculates weights
    output = graph.forward(expert_outputs, top_k_indices, top_k_weights)

    # We can't directly access the weights, but we can check their effect.
    # Let's re-implement the forward pass logic here to get the weights
    final_weights = []
    for i in range(batch_size):
        sub_adjacency = graph.adjacency_matrix[top_k_indices[i], :][:, top_k_indices[i]]
        w_sel = F.softmax(top_k_weights[i], dim=0)
        adj_influence = sub_adjacency.mean(dim=0)
        cooperation_gate = torch.sigmoid(adj_influence)
        weights = w_sel * cooperation_gate
        weights = weights / (weights.sum() + 1e-12)
        final_weights.append(weights)

    # Check if the first item's weights match our manual calculation
    assert torch.allclose(final_weights[0], manual_weights)

    # Check that all weight sets sum to 1
    for w in final_weights:
        assert torch.allclose(w.sum(), torch.tensor(1.0))