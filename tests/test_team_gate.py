import torch
import pytest
from models.moe import Gate


@pytest.mark.parametrize("mode", ["topk", "gumbel_topk"])
def test_team_gate_weights_sum_and_shape(mode):
    input_dim = 32
    num_experts = 8
    top_k = 3
    batch_size = 16

    gate = Gate(input_dim, num_experts, top_k, mode=mode, temperature=1.0)
    x = torch.randn(batch_size, input_dim, requires_grad=True)

    topk_idx, logits = gate(x)
    weights = gate.last_topk_weights

    assert topk_idx.shape == (batch_size, top_k)
    assert logits.shape == (batch_size, num_experts)
    assert weights is not None
    assert weights.shape == (batch_size, top_k)

    # Weights sum to ~1 per sample
    s = weights.sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


@pytest.mark.parametrize("mode", ["topk", "gumbel_topk"])
def test_team_gate_gradients_flow(mode):
    input_dim = 16
    num_experts = 6
    top_k = 2
    batch_size = 8

    gate = Gate(input_dim, num_experts, top_k, mode=mode, temperature=0.7)
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    topk_idx, logits = gate(x)

    # Simple scalar objective using logits (ensures gradients flow through gate network)
    loss = logits.sum()
    loss.backward()

    assert x.grad is not None


def test_entropy_fallback_adds_expert():
    input_dim = 32
    num_experts = 8
    top_k = 3
    batch_size = 16

    # High entropy threshold to ensure fallback is triggered
    gate = Gate(input_dim, num_experts, top_k, mode='topk', use_entropy_fallback=True, entropy_threshold=0.1)
    
    # Create input that results in high entropy (uniform logits)
    x = torch.zeros(batch_size, input_dim, requires_grad=True) # Zeros will produce uniform logits if bias is zero

    topk_idx, logits = gate(x)

    # Check that top_k + 1 experts are selected
    assert topk_idx.shape == (batch_size, top_k + 1)