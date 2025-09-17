import torch
from rl.rl_env import MoEGatingEnv
from rl.rl_agent import ReinforceAgent
from rl.rl_runner import RLRunner
from models.moe import Expert

class DummyModel(torch.nn.Module):
    def __init__(self, n, fd, nc):
        super().__init__(); self.experts=torch.nn.ModuleList([Expert(fd,nc) for _ in range(n)])
    def forward(self,x):
        outs=[e(x) for e in self.experts]
        return torch.stack(outs,dim=1).mean(dim=1), None

def test_rl_runner_ema_progression(tmp_path):
    torch.manual_seed(0)
    num_experts=3; feature_dim=6; num_classes=4
    model=DummyModel(num_experts, feature_dim, num_classes)
    env=MoEGatingEnv(model, num_experts, feature_dim, num_classes, top_k=2, batch_size=4)
    agent=ReinforceAgent(env.observation_dim, num_experts)
    runner=RLRunner(env, agent, episodes=8, out_dir=str(tmp_path))
    rows=runner.run(log_every=100)
    ema_values=[r['ema_reward'] for r in rows]
    # EMA should be monotonic smoothing toward recent rewards, first equals first reward
    assert ema_values[0]==rows[0]['reward']
    # Ensure at least one later EMA differs from first (evidence of update)
    assert any(abs(ema_values[i]-ema_values[0])>1e-6 for i in range(1,len(ema_values)))
    # moving avg reward should be defined and within min/max of seen rewards window
    for r in rows:
        assert r['moving_avg_reward'] is not None
        assert min(rr['reward'] for rr in rows) - 1e-6 <= r['moving_avg_reward'] <= max(rr['reward'] for rr in rows) + 1e-6
