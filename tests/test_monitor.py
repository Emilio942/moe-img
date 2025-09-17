import time
import torch
import torch.nn as nn
from monitor.probe import SystemProbe

# A simple dummy model for testing
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    def forward(self, x):
        return self.linear(x)

def test_time_measurement():
    """Tests if the time measurement is working correctly."""
    probe = SystemProbe()
    model = DummyModel()
    inputs = torch.randn(1, 10)
    probe.start(model, inputs)
    time.sleep(0.05)  # 50ms
    measurements = probe.stop()
    assert "time_ms" in measurements
    assert 50 <= measurements["time_ms"] <= 70  # Allow for some overhead

def test_memory_measurement():
    """Tests if the memory measurement is working correctly."""
    probe = SystemProbe()
    model = DummyModel()
    inputs = torch.randn(1, 10)
    probe.start(model, inputs)
    # Allocate a 10MB tensor
    tensor = torch.randn(10 * 1024 * 1024 // 4) # 10MB of floats
    measurements = probe.stop()
    assert "mem_mb" in measurements
    # The memory increase should be around 10MB
    # It might not be exactly 10MB due to other allocations
    assert 9.5 <= measurements["mem_mb"] <= 12
