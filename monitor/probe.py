import time
import psutil
import os
import torch
from thop import profile


class SystemProbe:
    """Simple runtime probe to gather timing, memory, FLOPs (optional) and an energy proxy.

    energy_mj = flops * energy_coeff + time_ms * time_coeff + mem_mb * mem_coeff
    Coefficients are configurable so tests can rely on relative ordering only.
    """

    def __init__(self, measure_flops: bool = True, energy_coeff: float = 1e-9,
                 time_coeff: float = 1e-6, mem_coeff: float = 1e-3):
        self.process = psutil.Process(os.getpid())
        self._start_time = None
        self._start_mem = None
        self.model = None
        self.inputs = None
        self.measure_flops = measure_flops
        # Coefficients for the energy proxy
        self.energy_coeff = energy_coeff
        self.time_coeff = time_coeff
        self.mem_coeff = mem_coeff

    def start(self, model, inputs):
        self._start_time = time.perf_counter()
        self._start_mem = self.process.memory_info().rss
        self.model = model
        self.inputs = inputs

    def stop(self):
        if self._start_time is None or self._start_mem is None:
            return {}

        end_time = time.perf_counter()
        end_mem = self.process.memory_info().rss

        time_ms = (end_time - self._start_time) * 1000.0
        mem_mb = (end_mem - self._start_mem) / (1024 * 1024)
        if mem_mb < 0:
            mem_mb = 0.0  # clamp

        flops = 0
        energy_mj = 0.0
        if self.measure_flops and self.model is not None and self.inputs is not None:
            try:
                flops, _ = profile(self.model, inputs=(self.inputs,), verbose=False)
                energy_mj = flops * self.energy_coeff
            except Exception:
                pass

        # Add time + memory contributions regardless of FLOPs success
        energy_mj += max(time_ms, 0.0) * self.time_coeff + max(mem_mb, 0.0) * self.mem_coeff

        # Reset start markers to avoid accidental reuse
        self._start_time = None
        self._start_mem = None
        self.model = None
        self.inputs = None

        return {
            'time_ms': time_ms,
            'mem_mb': mem_mb,
            'flops': flops,
            'energy_mj': energy_mj
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False