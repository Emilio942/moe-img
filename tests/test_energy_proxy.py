import torch
from monitor.probe import SystemProbe

class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(16, 16)
    def forward(self, x):
        return self.lin(x)

class LargeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(8):
            layers.append(torch.nn.Linear(128,128))
            layers.append(torch.nn.ReLU())
        self.seq = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.seq(x)

def run_probe(model, shape):
    x = torch.randn(*shape)
    probe = SystemProbe(measure_flops=True)
    probe.start(model, x)
    with torch.no_grad():
        _ = model(x)
    m = probe.stop()
    return m['energy_mj']


def test_energy_proxy_scales_with_workload():
    e_small = run_probe(SmallModel(), (32,16))
    e_large = run_probe(LargeModel(), (32,128))
    # Larger model + larger input dim should yield higher energy proxy
    assert e_large > e_small * 1.5  # relaxed factor: larger workload should cost more
    assert e_small > 0


def test_energy_proxy_rank_correlation_across_batchsizes():
    # Keep model fixed, increase batch size -> energy proxy should not decrease
    model = LargeModel()
    e_b32 = run_probe(model, (32,128))
    e_b64 = run_probe(model, (64,128))
    assert e_b64 >= e_b32 * 0.9  # allow some noise, but should be roughly monotonic


def test_energy_proxy_component_contributions():
    # Verify that if FLOPs estimation fails, time/mem terms still contribute positively
    class NoFlopsModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(10, 10))
        def forward(self, x):
            return x @ self.w  # thop may still profile, but we wrap with measure_flops=False to force fallback

    x = torch.randn(32, 10)
    probe = SystemProbe(measure_flops=False)
    m = NoFlopsModel()
    probe.start(m, x)
    with torch.no_grad():
        _ = m(x)
    res = probe.stop()
    assert res['energy_mj'] > 0
