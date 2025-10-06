import torch
from rl.rl_env import MoEGatingEnv
from models.moe import Expert


class DummyModel(torch.nn.Module):
    def __init__(self, n=3, fd=6, nc=4):
        super().__init__()
        self.experts = torch.nn.ModuleList([Expert(fd, nc) for _ in range(n)])

    def forward(self, x):
        outs = [e(x) for e in self.experts]
        return torch.stack(outs, dim=1).mean(dim=1), None


def test_env_uses_normalized_cost_when_enabled():
    torch.manual_seed(0)
    num_experts, feature_dim, num_classes = 3, 6, 4
    model = DummyModel(n=num_experts, fd=feature_dim, nc=num_classes)
    env = MoEGatingEnv(model, num_experts, feature_dim, num_classes, top_k=2, batch_size=4, use_normalizer=True)

    # Ensure cost model expects normalized inputs
    assert getattr(env.cost_model, 'input_mode', None) == 'normalized'

    obs, _ = env.reset()
    action = [0, 1]
    _, reward, terminated, truncated, info = env.step(action)
    assert terminated is True and truncated is False
    assert isinstance(reward, float)
    # Info should include raw and normalized measurements for diagnostics
    assert 'measurements_raw' in info and 'measurements_norm' in info
    mr, mn = info['measurements_raw'], info['measurements_norm']
    for k in ('time_ms', 'mem_mb', 'flops'):
        assert k in mr and k in mn


def test_env_raw_mode_without_normalizer():
    torch.manual_seed(0)
    num_experts, feature_dim, num_classes = 3, 6, 4
    model = DummyModel(n=num_experts, fd=feature_dim, nc=num_classes)
    env = MoEGatingEnv(model, num_experts, feature_dim, num_classes, top_k=2, batch_size=4, use_normalizer=False)

    assert getattr(env.cost_model, 'input_mode', None) == 'raw'
    env.reset()
    _, reward, _, _, info = env.step([1, 2])
    assert isinstance(reward, float)
    # In raw mode, we still surface measurements_norm equal to measurements_raw (env sets norm to raw when normalizer off)
    assert 'measurements_raw' in info and 'measurements_norm' in info
    assert info['measurements_norm'] == info['measurements_raw']
