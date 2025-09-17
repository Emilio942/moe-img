import torch
from models.moe import MoEModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_budget():
    """Tests if the parameter increase due to the graph is within a budget (e.g., 10%)."""
    # Hyperparameters
    input_dim = 32
    output_dim = 10
    num_experts = 8
    top_k = 2
    budget_increase_threshold = 0.10  # 10%

    # --- Phase 2 Style Model (No Graph) ---
    # We simulate this by not counting the graph parameters
    model_with_graph = MoEModel(input_dim, output_dim, num_experts, top_k)
    
    # Parameters of the graph
    graph_params = count_parameters(model_with_graph.expert_graph)
    
    # Total parameters minus the graph
    base_params = count_parameters(model_with_graph) - graph_params

    # --- Phase 3 Style Model (With Graph) ---
    total_params = count_parameters(model_with_graph)

    # Calculate the increase
    increase = total_params - base_params
    increase_ratio = increase / base_params

    print(f"\nBase parameters (Phase 2): {base_params}")
    print(f"Graph parameters: {graph_params}")
    print(f"Total parameters (Phase 3): {total_params}")
    print(f"Parameter increase ratio: {increase_ratio:.4f}")

    assert increase_ratio <= budget_increase_threshold, f"Parameter increase {increase_ratio:.2f} exceeds budget {budget_increase_threshold:.2f}"
