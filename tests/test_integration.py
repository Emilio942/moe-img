
import torch
from models.moe import MoEModel

def test_moe_model_deterministic_with_graph():
    """Tests if the MoEModel forward pass is deterministic with a fixed seed, including graph aggregation."""
    # Hyperparameters
    input_dim = 32
    output_dim = 10
    num_experts = 8
    top_k = 2
    batch_size = 4
    seed = 42

    # Create a single input tensor to be used for both runs
    torch.manual_seed(seed)
    input_tensor = torch.randn(batch_size, input_dim)

    # --- First run ---
    torch.manual_seed(seed)
    model1 = MoEModel(input_dim, output_dim, num_experts, top_k)
    model1.use_adjacency = True # Ensure the graph is used
    model1.eval()
    output1, _ = model1(input_tensor)

    # --- Second run ---
    torch.manual_seed(seed)
    model2 = MoEModel(input_dim, output_dim, num_experts, top_k)
    model2.use_adjacency = True # Ensure the graph is used
    model2.eval()
    output2, _ = model2(input_tensor)

    # Assert that the two outputs are identical
    assert torch.allclose(output1, output2), "Model output is not deterministic with a fixed seed."
