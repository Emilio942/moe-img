import csv
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch

from rl.rl_env import MoEGatingEnv
from rl.rl_agent import ReinforceAgent


def _compute_pareto(points):
    """Compute Pareto front for (cost, accuracy) pairs (min cost, max acc).
    points: list of dicts with 'cost' and 'accuracy'. Returns subset sorted by cost.
    """
    # Sort by cost asc, accuracy desc
    sorted_pts = sorted(points, key=lambda d: (d['cost'], -d['accuracy']))
    pareto = []
    best_acc = -1.0
    for p in sorted_pts:
        if p['accuracy'] > best_acc:
            pareto.append(p)
            best_acc = p['accuracy']
    return pareto

class RLRunner:
    def __init__(self, env: MoEGatingEnv, agent: ReinforceAgent, episodes: int = 50, device=None, out_dir: str = 'reports/rl_runs', run_name: Optional[str] = None, monitor_log_path: Optional[str] = None, monitor_compare_with: Optional[str] = None):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.device = device or torch.device('cpu')
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        self.run_name = run_name or f'run_{ts}'
        self.csv_path = self.out_dir / f'{self.run_name}.csv'
        self.jsonl_path = self.out_dir / f'{self.run_name}.jsonl'
        self.meta_path = self.out_dir / f'{self.run_name}_meta.json'
        self.rows: List[Dict[str, Any]] = []
        # EMA + rolling metrics
        self._ema_reward: Optional[float] = None
        self._ema_entropy: Optional[float] = None
        self.ema_alpha: float = 0.1
        self.window: int = 5
        self._recent_rewards: List[float] = []
        # Monitor logging for time/overhead comparison
        # Default log path infers on/off from env sampling behavior
        is_on = True
        try:
            is_on = getattr(self.env, 'sample_every', 1) == 1 and getattr(getattr(self.env, 'probe', None), 'measure_flops', True)
        except Exception:
            pass
        default_mon_name = f"monitor_{'on' if is_on else 'off'}_{self.run_name}.jsonl"
        self.monitor_log_path = Path(monitor_log_path or os.environ.get('RL_MONITOR_LOG_PATH', str(self.out_dir / default_mon_name)))
        self.monitor_compare_with = monitor_compare_with or os.environ.get('RL_MONITOR_COMPARE_WITH')
        self._episode_times_ms: List[float] = []

    def run(self, log_every: int = 1):
        state, _ = self.env.reset()
        state_t = torch.tensor(state, dtype=torch.float32)
        for ep in range(1, self.episodes + 1):
            t0 = time.perf_counter()
            action = self.agent.select_action(state_t)
            # Convert sampled scalar action into repeated list for multi-discrete if needed
            if hasattr(self.env.action_space, 'nvec') and len(self.env.action_space.nvec) > 1:
                action_vec = [action] * len(self.env.action_space.nvec)  # replicate across slots
            else:
                action_vec = action
            next_state, reward, terminated, truncated, info = self.env.step(action_vec)
            # Per-episode wall time
            t1 = time.perf_counter()
            self._episode_times_ms.append((t1 - t0) * 1000.0)
            self.agent.rewards.append(reward)
            self.agent.finish_episode()
            state_t = torch.tensor(next_state, dtype=torch.float32)

            entropy = self.agent.entropies[-1] if self.agent.entropies else None
            # Update EMAs & rolling window
            self._ema_reward = reward if self._ema_reward is None else (1 - self.ema_alpha) * self._ema_reward + self.ema_alpha * reward
            if entropy is not None:
                self._ema_entropy = entropy if self._ema_entropy is None else (1 - self.ema_alpha) * self._ema_entropy + self.ema_alpha * entropy
            self._recent_rewards.append(reward)
            if len(self._recent_rewards) > self.window:
                self._recent_rewards.pop(0)
            moving_avg_reward = sum(self._recent_rewards) / len(self._recent_rewards)

            row = {
                'episode': ep,
                'reward': reward,
                'accuracy': info.get('accuracy'),
                'cost': info.get('cost'),
                'action': info.get('action'),
                'entropy': entropy,
                'ema_reward': self._ema_reward,
                'ema_entropy': self._ema_entropy,
                'moving_avg_reward': moving_avg_reward,
            }
            self.rows.append(row)
            if ep % log_every == 0:
                print(f"[RLRunner] Ep {ep}/{self.episodes} reward={reward:.4f} acc={row['accuracy']:.4f} cost={row['cost']:.4f}")
            if terminated or truncated:
                state, _ = self.env.reset()
                state_t = torch.tensor(state, dtype=torch.float32)
        # After episodes loop
        self._write_outputs()
        self._write_monitor_log()
        self._maybe_compare_overhead()
        return self.rows

    def _write_outputs(self):
        headers = ['episode','reward','accuracy','cost','action','entropy','ema_reward','ema_entropy','moving_avg_reward']
        with self.csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in self.rows:
                writer.writerow(r)
        with self.jsonl_path.open('w') as f:
            for r in self.rows:
                f.write(json.dumps(r) + '\n')
        pareto = _compute_pareto(self.rows)
        meta = {
            'episodes': self.episodes,
            'pareto': pareto,
            'csv': str(self.csv_path),
            'jsonl': str(self.jsonl_path)
        }
        with self.meta_path.open('w') as f:
            json.dump(meta, f, indent=2)

    def _write_monitor_log(self):
        # Write per-episode timing as JSONL compatible with compare script (use episode as epoch)
        self.monitor_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.monitor_log_path.open('w', encoding='utf-8') as f:
            for idx, t in enumerate(self._episode_times_ms, start=1):
                rec = {
                    'epoch': idx,
                    'avg_time_ms': t,
                    'avg_mem_mb': None,
                    'avg_energy_mj': None,
                }
                f.write(json.dumps(rec) + '\n')

    def _maybe_compare_overhead(self):
        other = self.monitor_compare_with
        if not other or not os.path.exists(other):
            return
        try:
            from reports.compare_monitor_overhead import plot_overhead
            # Heuristic: if our log name contains 'off' then treat other as 'on'
            if 'off' in self.monitor_log_path.name:
                summary = plot_overhead(on_log_path=other, off_log_path=str(self.monitor_log_path), out_dir=str(self.out_dir))
            else:
                summary = plot_overhead(on_log_path=str(self.monitor_log_path), off_log_path=other, out_dir=str(self.out_dir))
            print('[RLRunner] Monitor overhead summary:', json.dumps(summary))
        except Exception as e:
            print('[RLRunner] Overhead comparison failed:', e)

__all__ = ['RLRunner', '_compute_pareto']
