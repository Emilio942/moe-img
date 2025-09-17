import torch, os
from rl.rl_env import MoEGatingEnv
from rl.rl_agent import ReinforceAgent
from rl.rl_runner import RLRunner
from rl import rl_analysis
from models.moe import Expert

def test_rl_analysis_generates_plots(tmp_path):
    torch.manual_seed(0)
    # Small dummy setup
    num_experts=3; feature_dim=6; num_classes=4
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.experts=torch.nn.ModuleList([Expert(feature_dim,num_classes) for _ in range(num_experts)])
        def forward(self,x):
            outs=[e(x) for e in self.experts]; return torch.stack(outs,dim=1).mean(dim=1), None
    model=Dummy()
    env=MoEGatingEnv(model, num_experts, feature_dim, num_classes, top_k=2, batch_size=4)
    agent=ReinforceAgent(env.observation_dim, num_experts)
    runner=RLRunner(env, agent, episodes=5, out_dir=str(tmp_path))
    rows=runner.run(log_every=100)
    assert len(rows)==5
    csv_files=[p for p in tmp_path.iterdir() if p.suffix=='.csv']
    assert csv_files
    out_plot_dir=tmp_path / 'plots'
    rl_analysis.generate_all(str(csv_files[0]), str(out_plot_dir))
    # Expect at least reward & pareto plots
    expected=['reward_curve.png','pareto.png']
    for name in expected:
        assert (out_plot_dir / name).exists(), f"Missing plot {name}"
