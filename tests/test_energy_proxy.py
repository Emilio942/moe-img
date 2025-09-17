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
