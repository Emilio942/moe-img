
import torch
from models.moe import Gate

def test_gate_selection():
    """Tests the gate's output shape and index range."""
    # Hyperparameters
    input_dim = 32
    num_experts = 8
    top_k = 2
    batch_size = 16

    # Initialize gate
    gate = Gate(input_dim, num_experts, top_k)

    # Create dummy input
    dummy_input = torch.randn(batch_size, input_dim)

    # Forward pass
    top_k_indices, logits = gate(dummy_input)

    # 1. Check shape of the output indices
    assert top_k_indices.shape == (batch_size, top_k), f"Expected shape {(batch_size, top_k)}, but got {top_k_indices.shape}"

    # 2. Check if indices are of integer type
    assert top_k_indices.dtype == torch.long, f"Expected dtype torch.long, but got {top_k_indices.dtype}"

    # 3. Check if all indices are within the valid range [0, num_experts)
    assert torch.all(top_k_indices >= 0), "Found an index less than 0"
    assert torch.all(top_k_indices < num_experts), f"Found an index greater than or equal to {num_experts}"

    # 4. Check shape of the logits
    assert logits.shape == (batch_size, num_experts), f"Expected shape {(batch_size, num_experts)}, but got {logits.shape}"
