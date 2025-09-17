import torch
import numpy as np
from rl.rl_env import MoEGatingEnv
from models.moe import Expert

class DummyModel(torch.nn.Module):
    def __init__(self, num_experts, feature_dim, num_classes):
        super().__init__()
        self.experts = torch.nn.ModuleList([
            Expert(feature_dim, num_classes) for _ in range(num_experts)
        ])
    def forward(self, x):
        # Not used directly in env when experts attribute present
        outs = [e(x) for e in self.experts]
        return torch.stack(outs, dim=1).mean(dim=1), None

def test_env_normalizer_updates_state():
    torch.manual_seed(0)
    num_experts = 4
    feature_dim = 8
    num_classes = 5
    model = DummyModel(num_experts, feature_dim, num_classes)
    env = MoEGatingEnv(model, num_experts, feature_dim, num_classes, top_k=2, use_normalizer=True)
    # Run a couple of steps to update normalizer EMA
    _, _ = env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        env.step(action)
    norm = env.normalizer
    state = norm.state_dict()
    stats = state['stats']
    # Means should be non-zero for at least one component
    moved = any(abs(stats[k]['mean']) > 0 for k in stats)
    assert moved, "Normalizer means did not update"
    # Variance should be >=0 and positive for at least one component after multiple updates
    var_positive = any(stats[k]['var'] > 0 for k in stats)
    assert var_positive, "Normalizer variances did not update"
