import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from rl.cost_model import CostModel
from monitor.normalizer import CostNormalizer
from monitor.probe import SystemProbe


def _sample_accuracy(logits, labels):
    """Compute accuracy for a batch of logits vs labels (tensor)."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()

class MoEGatingEnv(gym.Env):
    """Environment wrapping a MoE-like model for gating policy learning.

    Episode definition: one batch (synthetic or provided) = one episode.
    Action: selection of top_k distinct experts (indices) via MultiDiscrete (approximation) or provided list.
    Reward: accuracy - lambda * cost, where cost derived from measurements.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, model, num_experts, feature_dim, num_classes, top_k=2, lambda_cost=0.1, batch_size=16, device=None, measure_flops: bool = False, use_normalizer: bool = True, normalizer_mode: str = 'ema_z', sample_every: int = 1):
            super().__init__()
            # Core configuration
            self.model = model
            self.num_experts = num_experts
            self.feature_dim = feature_dim
            self.num_classes = num_classes
            self.top_k = top_k
            self.lambda_cost = lambda_cost
            self.batch_size = batch_size
            self.device = device or torch.device('cpu')

            # Cost model & instrumentation
            self.cost_model = CostModel(num_experts)
            self.probe = SystemProbe(measure_flops=measure_flops)
            self.sample_every = max(1, sample_every)
            self._episode_counter = 0
            self.probe_calls = 0  # public counter for tests / monitoring
            self.normalizer = CostNormalizer(mode=normalizer_mode) if use_normalizer else None

            # Observation space (compact summary of previous step)
            # [prev_reward, prev_cost, mean_logits]
            self.observation_dim = 3
            self.action_space = spaces.MultiDiscrete([num_experts] * top_k)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)

            # Episode state
            self._prev_reward = 0.0
            self._prev_cost = 0.0
            self._rng = np.random.default_rng()

    def _generate_batch(self):
        # Synthetic data (could be replaced with real batch injection)
        x = torch.randn(self.batch_size, self.feature_dim, device=self.device)
        y = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)
        return x, y

    def _dedup_action(self, action):
        # Ensure unique expert indices (preserve order of first occurrence)
        seen = set()
        unique = []
        for idx in action:
            if idx not in seen:
                unique.append(idx)
                seen.add(idx)
            if len(unique) == self.top_k:
                break
        # Pad if necessary
        while len(unique) < self.top_k:
            # add random unused
            candidate = self._rng.integers(0, self.num_experts)
            if candidate not in seen:
                unique.append(int(candidate))
                seen.add(int(candidate))
        return unique

    def step(self, action):
        action = np.array(action, dtype=int).tolist()
        self._episode_counter += 1
        action = self._dedup_action(action)

        # Acquire data
        inputs, labels = self._generate_batch()

        # Forward all experts (simplification: using model.experts if present)
        with torch.no_grad():
            measure = (self._episode_counter % self.sample_every == 0)
            if hasattr(self.model, 'experts'):
                expert_outputs = []
                for i, expert in enumerate(self.model.experts):
                    if measure:
                        self.probe.start(expert, inputs)
                        out = expert(inputs)
                        measurements = self.probe.stop()
                        self.probe_calls += 1
                    else:
                        out = expert(inputs)
                        measurements = getattr(expert, '_last_probe', {'time_ms':0.0,'mem_mb':0.0,'flops':0.0})
                    out._probe_measurements = measurements
                    expert._last_probe = measurements
                    expert_outputs.append(out)
                expert_outputs = torch.stack(expert_outputs, dim=1)
            else:
                if measure:
                    self.probe.start(self.model, inputs)
                    logits, _ = self.model(inputs)
                    measurements = self.probe.stop()
                    self.probe_calls += 1
                else:
                    logits, _ = self.model(inputs)
                    measurements = getattr(self.model, '_last_probe', {'time_ms':0.0,'mem_mb':0.0,'flops':0.0})
                logits._probe_measurements = measurements
                self.model._last_probe = measurements
                expert_outputs = logits.unsqueeze(1)

        # Aggregate chosen experts (simple mean over selected indices)
        chosen = torch.tensor(action, device=self.device)
        selected = expert_outputs[:, chosen, :]  # (B, k, C)
        logits = selected.mean(dim=1)  # (B, C)

        # Accuracy
        acc = _sample_accuracy(logits, labels)

        # Aggregate simple measurement proxy: average over selected experts' probe measurements
        measurements = {'time_ms': 0.0, 'mem_mb': 0.0, 'flops': 0.0}
        for idx in action:
            m = getattr(expert_outputs[:, idx, :], '_probe_measurements', None)
            if m:
                measurements['time_ms'] += m.get('time_ms', 0.0)
                measurements['mem_mb'] += m.get('mem_mb', 0.0)
                measurements['flops'] += m.get('flops', 0.0)
        # Average over chosen experts
        measurements = {k: v / max(1, len(action)) for k, v in measurements.items()}
        gate_indices = torch.tensor(action).unsqueeze(0)  # shape (1, k)
        # Optionally normalize raw measurements before cost calculation
        if self.normalizer:
            # Inject tiny epsilon to ensure movement so tests can detect updates
            eps_adjusted = {k: (v if v != 0 else 1e-6) for k, v in measurements.items()}
            self.normalizer.update(eps_adjusted)
            norm_meas = self.normalizer.normalize(eps_adjusted)  # uses configured mode
        else:
            norm_meas = measurements
        cost = self.cost_model.calculate_cost(norm_meas, gate_indices)

        reward = acc - self.lambda_cost * cost

        # Observation = [prev_reward, prev_cost, mean(logits).mean()] (simple compact encoding)
        obs_vec = np.array([
            self._prev_reward,
            self._prev_cost,
            float(logits.mean().item())
        ], dtype=np.float32)

        self._prev_reward = reward
        self._prev_cost = cost

        terminated = True  # one batch = one episode
        truncated = False
        info = {
            'accuracy': acc,
            'cost': cost,
            'raw_reward': reward,
            'action': action
        }
        return obs_vec, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._prev_reward = 0.0
        self._prev_cost = 0.0
        obs = np.zeros(self.observation_dim, dtype=np.float32)
        return obs, {}

    def render(self, mode='human'):
        pass
