import torch
import torch.nn as nn
from models.expert_graph import ExpertGraph

class Expert(nn.Module):
    """A simple MLP expert."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class Gate(nn.Module):
    """A simple gate that selects the top-k experts."""
    def __init__(self, input_dim, num_experts, top_k, mode: str = 'topk', temperature: float = 1.0, use_entropy_fallback: bool = False, entropy_threshold: float = 0.8):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.mode = mode  # 'topk' or 'gumbel_topk'
        self.temperature = temperature
        self.use_entropy_fallback = use_entropy_fallback
        self.entropy_threshold = entropy_threshold
        self.net = nn.Linear(input_dim, num_experts)
        # Will be populated on forward for downstream use (optional)
        self.last_topk_weights = None

    def forward(self, x):
        """Returns top-k indices and the raw logits. Stores weights in self.last_topk_weights."""
        logits = self.net(x)
        k = self.top_k

        if self.use_entropy_fallback:
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            if entropy.mean() > self.entropy_threshold:
                k = min(self.top_k + 1, self.num_experts)

        if self.mode == 'gumbel_topk':
            # Differentiable Top-k approximation via Gumbel noise + softmax over selected
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
            scores = (logits + gumbel_noise) / max(self.temperature, 1e-6)
            topk_scores, top_k_indices = torch.topk(scores, k, dim=-1)
            # Soft weights over selected scores
            weights = torch.softmax(topk_scores, dim=-1)
            self.last_topk_weights = weights
        else:
            # Hard top-k selection, uniform weights on selected experts
            _, top_k_indices = torch.topk(logits, k, dim=-1)
            self.last_topk_weights = torch.full((*top_k_indices.shape,), 1.0 / k, device=logits.device, dtype=logits.dtype)
        return top_k_indices, logits

class VectorMoE(nn.Module):
    """A MoE model for vector inputs, combining Gate, Experts, and ExpertGraph."""
    def __init__(self, input_dim, output_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = Gate(input_dim, num_experts, top_k, use_entropy_fallback=True)
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.expert_graph = ExpertGraph(num_experts, output_dim, aggregation_type='weighted_sum')
        self.use_adjacency = True

    def forward(self, x, gate_indices=None):
        # Get expert selection from the gate if not provided
        if gate_indices is None:
            gate_indices, gate_logits = self.gate(x)
        else:
            gate_logits = None # Can't get logits if we bypass the gate

        # In a real scenario, we'd only compute the selected experts for efficiency.
        # For this example, we compute all and then select.
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Aggregate the expert outputs using the graph
        topk_weights = getattr(self.gate, 'last_topk_weights', None)
        final_output = self.expert_graph(
            expert_outputs,
            gate_indices,
            top_k_weights=topk_weights,
            use_adjacency=self.use_adjacency,
        )

        return final_output, gate_logits

class CifarMoE(nn.Module):
    """A CNN-based MoE model for CIFAR-10."""
    def __init__(self, num_experts, top_k, num_classes=10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 4x4
            nn.Flatten(),
        )
        # Calculate the flattened feature dimension
        feature_dim = 128 * 4 * 4
        self.head = VectorMoE(feature_dim, num_classes, num_experts, top_k)

    def forward(self, x):
        features = self.body(x)
        return self.head(features)


class MoEModel(VectorMoE):
    """
    Backwards-compatible alias used by training script and tests.

    Signature: MoEModel(input_dim, output_dim, num_experts, top_k)
    Forward returns: (final_output, gate_logits)
    """
    pass