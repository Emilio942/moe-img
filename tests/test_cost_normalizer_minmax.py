from monitor.normalizer import CostNormalizer


def test_minmax_mode_basic_range():
    norm = CostNormalizer(mode='minmax')
    seq = [
        {'time_ms': 2.0, 'mem_mb': 10.0, 'flops': 100.0},
        {'time_ms': 4.0, 'mem_mb': 20.0, 'flops': 200.0},
        {'time_ms': 3.0, 'mem_mb': 15.0, 'flops': 150.0},
    ]
    for m in seq:
        norm.update(m)
    normalized = norm.normalize({'time_ms': 3.0, 'mem_mb': 15.0, 'flops': 150.0})
    for v in normalized.values():
        assert 0.0 <= v <= 1.0
    # Edge values map to 0 and 1
    n_min = norm.normalize({'time_ms': 2.0, 'mem_mb': 10.0, 'flops': 100.0})
    n_max = norm.normalize({'time_ms': 4.0, 'mem_mb': 20.0, 'flops': 200.0})
    assert abs(n_min['time_ms'] - 0.0) < 1e-9 and abs(n_max['time_ms'] - 1.0) < 1e-9


def test_switching_modes_independence():
    # Build some stats with ema_z first
    ema_norm = CostNormalizer(mode='ema_z', beta=0.5)
    for v in [1.0, 2.0, 3.0, 4.0]:
        ema_norm.update({'time_ms': v, 'mem_mb': v*10, 'flops': v*100})
    z_val = ema_norm.normalize({'time_ms': 5.0, 'mem_mb': 50.0, 'flops': 500.0})['time_ms']

    # New normalizer in minmax mode sees independent stats
    mm_norm = CostNormalizer(mode='minmax')
    for v in [1.0, 2.0, 3.0, 4.0]:
        mm_norm.update({'time_ms': v, 'mem_mb': v*10, 'flops': v*100})
    mm_val = mm_norm.normalize({'time_ms': 5.0, 'mem_mb': 50.0, 'flops': 500.0})['time_ms']

    # Ensure both produce finite distinct transformations (not identical by coincidence likely)
    assert z_val != mm_val
    assert all(key in mm_norm.state_dict()['mins'] for key in ['time_ms','mem_mb','flops'])


def test_state_roundtrip_mode_field():
    mm_norm = CostNormalizer(mode='minmax')
    mm_norm.update({'time_ms': 2.5, 'mem_mb': 11.0, 'flops': 120.0})
    sd = mm_norm.state_dict()
    restored = CostNormalizer()
    restored.load_state_dict(sd)
    assert restored.mode == 'minmax'
    assert restored.mins['time_ms'] == mm_norm.mins['time_ms']
