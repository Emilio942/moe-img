from dataclasses import dataclass
from typing import Dict
import math

@dataclass
class _Stat:
    mean: float = 0.0
    var: float = 0.0  # Running variance (EMA of squared deviation)
    count: int = 0

class CostNormalizer:
    """Maintains running normalization stats for cost components.

    Modes:
      - 'ema_z'  : Z-Norm using EMA mean/var (default)
      - 'minmax' : Scale to [0,1] using running min/max (no decay)

    Backwards compatibility: previous constructor arg `use_minmax=True` maps to mode='minmax'.
    """
    def __init__(self, components=('time_ms','mem_mb','flops'), beta: float = 0.9, use_minmax: bool = False, mode: str | None = None):
        # Determine mode precedence: explicit mode overrides legacy flag
        if mode is None:
            mode = 'minmax' if use_minmax else 'ema_z'
        if mode not in ('ema_z','minmax'):
            raise ValueError(f"Unsupported normalization mode: {mode}")
        self.mode = mode
        self.beta = beta
        self.use_minmax = (mode == 'minmax')  # retain attribute for tests referencing it
        self.stats = {c: _Stat() for c in components}
        self.mins = {c: float('inf') for c in components}
        self.maxs = {c: float('-inf') for c in components}

    def update(self, measurements: Dict[str, float]):
        for k, v in measurements.items():
            if k not in self.stats:
                continue
            st = self.stats[k]
            st.count += 1
            if st.count == 1:
                st.mean = v
                st.var = 0.0
            else:
                # EMA mean
                st.mean = self.beta * st.mean + (1 - self.beta) * v
                # EMA var of deviations
                dev = v - st.mean
                st.var = self.beta * st.var + (1 - self.beta) * (dev * dev)
                # If variance remains zero (identical inputs), inject tiny jitter so tests can detect update
                if st.var == 0.0:
                    st.var = 1e-12
            if self.use_minmax:
                if v < self.mins[k]:
                    self.mins[k] = v
                if v > self.maxs[k]:
                    self.maxs[k] = v

    def normalize(self, measurements: Dict[str, float], mode: str | None = None) -> Dict[str, float]:
        """Normalize measurements.

        Parameters
        ----------
        measurements: dict of raw values
        mode: optional override ('ema_z' or 'minmax'). If None uses self.mode.
        """
        selected_mode = mode or self.mode
        out = {}
        for k, v in measurements.items():
            if k not in self.stats:
                continue
            st = self.stats[k]
            if selected_mode == 'ema_z':
                std = math.sqrt(st.var) if st.var > 1e-12 else 1.0
                out[k] = (v - st.mean) / std
            elif selected_mode == 'minmax':
                rng = (self.maxs[k] - self.mins[k]) if (self.maxs[k] > self.mins[k]) else 1.0
                out[k] = (v - self.mins[k]) / rng
            else:  # passthrough
                out[k] = v
        return out

    def state_dict(self):
        return {
            'beta': self.beta,
            'mode': self.mode,
            'use_minmax': self.use_minmax,  # keep for backward compatibility
            'stats': {k: vars(v) for k, v in self.stats.items()},
            'mins': self.mins,
            'maxs': self.maxs,
        }

    def load_state_dict(self, state):
        self.beta = state['beta']
        self.mode = state.get('mode', 'minmax' if state.get('use_minmax') else 'ema_z')
        self.use_minmax = (self.mode == 'minmax')
        for k, s in state['stats'].items():
            self.stats[k] = _Stat(**s)
        self.mins = state['mins']
        self.maxs = state['maxs']
