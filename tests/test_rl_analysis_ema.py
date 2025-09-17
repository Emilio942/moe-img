import torch, os
from rl.rl_env import MoEGatingEnv
from rl.rl_agent import ReinforceAgent
from rl.rl_runner import RLRunner
from rl import rl_analysis
from models.moe import Expert

class Dummy(torch.nn.Module):
    def __init__(self, n=3, fd=6, nc=4):
        super().__init__(); self.experts=torch.nn.ModuleList([Expert(fd,nc) for _ in range(n)])
    def forward(self,x):
        outs=[e(x) for e in self.experts]; return torch.stack(outs,dim=1).mean(dim=1), None

def test_rl_analysis_ema_plots(tmp_path):
    torch.manual_seed(0)
    model=Dummy()
    env=MoEGatingEnv(model, 3, 6, 4, top_k=2, batch_size=4)
    agent=ReinforceAgent(env.observation_dim, 3)
    runner=RLRunner(env, agent, episodes=6, out_dir=str(tmp_path))
    runner.run(log_every=100)
    csv_files=[p for p in tmp_path.iterdir() if p.suffix=='.csv']
    out_plot_dir=tmp_path / 'plots'
    rl_analysis.generate_all(str(csv_files[0]), str(out_plot_dir))
    # Extended metrics should produce additional plots
    for name in ['ema_reward_curve.png','moving_avg_reward_curve.png']:
        assert (out_plot_dir / name).exists(), f"Missing {name}"
