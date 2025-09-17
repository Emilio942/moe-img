from monitor.normalizer import CostNormalizer

def test_cost_normalizer_updates_and_normalizes():
    norm = CostNormalizer(beta=0.5, use_minmax=True)
    measurements_seq = [
        {'time_ms': 10.0, 'mem_mb': 100.0, 'flops': 1000.0},
        {'time_ms': 12.0, 'mem_mb': 120.0, 'flops': 1100.0},
        {'time_ms': 11.0, 'mem_mb': 130.0, 'flops': 900.0},
    ]
    for m in measurements_seq:
        norm.update(m)
    z = norm.normalize({'time_ms': 11.0, 'mem_mb': 125.0, 'flops': 950.0}, mode='z')
    assert all(k in z for k in ['time_ms','mem_mb','flops'])
    minmax = norm.normalize({'time_ms': 11.0, 'mem_mb': 125.0, 'flops': 950.0}, mode='minmax')
    # min/max scaling should be between 0 and 1
    assert 0.0 <= minmax['time_ms'] <= 1.0
    assert 0.0 <= minmax['mem_mb'] <= 1.0
    assert 0.0 <= minmax['flops'] <= 1.0

def test_cost_normalizer_state_roundtrip():
    norm = CostNormalizer()
    norm.update({'time_ms': 5.0, 'mem_mb': 50.0, 'flops': 500.0})
    sd = norm.state_dict()
    new_norm = CostNormalizer()
    new_norm.load_state_dict(sd)
    assert new_norm.stats['time_ms'].mean == norm.stats['time_ms'].mean
