import torch
from rl.rl_env import MoEGatingEnv
from rl.rl_agent import ReinforceAgent
from rl.rl_runner import RLRunner
from models.moe import Expert

class Dummy(torch.nn.Module):
    def __init__(self, n=3, fd=6, nc=4):
        super().__init__(); self.experts=torch.nn.ModuleList([Expert(fd,nc) for _ in range(n)])
    def forward(self,x):
        outs=[e(x) for e in self.experts]; return torch.stack(outs,dim=1).mean(dim=1), None

def test_rl_agent_stability_trend(tmp_path):
    torch.manual_seed(0)
    model=Dummy()
    env=MoEGatingEnv(model,3,6,4, top_k=2, batch_size=8)
    agent=ReinforceAgent(env.observation_dim,3)
    # Run more episodes to allow trend emergence
    episodes=25
    runner=RLRunner(env, agent, episodes=episodes, out_dir=str(tmp_path))
    rows=runner.run(log_every=10)
    # Collect ema rewards (ignore None)
    ema=[r['ema_reward'] for r in rows if r['ema_reward'] is not None]
    assert len(ema) == episodes  # all should be set
    # Basic stability checks:
    # 1. EMA within observed reward bounds
    rewards=[r['reward'] for r in rows]
    mn, mx = min(rewards), max(rewards)
    assert all(mn - 1e-6 <= v <= mx + 1e-6 for v in ema)
    # 2. Stability: variance should be > 0 (learning signal present)
    mean_ema = sum(ema)/len(ema)
    var_ema = sum((v-mean_ema)**2 for v in ema)/len(ema)
    assert var_ema > 1e-4, "EMA reward variance too low (no learning dynamics)"
    # 3. No catastrophic collapse: min not far below last value relative to range
    ema_min, ema_max = min(ema), max(ema)
    if ema_max - ema_min > 1e-6:
        rel_pos_last = (ema[-1]-ema_min)/(ema_max-ema_min)
        assert rel_pos_last > 0.1, f"Final EMA near global minimum (collapse): pos={rel_pos_last:.3f}"
