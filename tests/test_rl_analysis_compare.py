import torch
from rl.rl_env import MoEGatingEnv
from rl.rl_agent import ReinforceAgent
from rl.rl_runner import RLRunner
from rl.rl_analysis import compare_runs
from models.moe import Expert

class Dummy(torch.nn.Module):
    def __init__(self, n=3, fd=6, nc=4):
        super().__init__(); self.experts=torch.nn.ModuleList([Expert(fd,nc) for _ in range(n)])
    def forward(self,x):
        outs=[e(x) for e in self.experts]; return torch.stack(outs,dim=1).mean(dim=1), None

def test_compare_runs_outputs(tmp_path):
    torch.manual_seed(0)
    # Create two runs
    paths=[]
    for seed in [0,1]:
        torch.manual_seed(seed)
        model=Dummy()
        env=MoEGatingEnv(model,3,6,4, top_k=2, batch_size=4)
        agent=ReinforceAgent(env.observation_dim,3)
        runner=RLRunner(env, agent, episodes=5, out_dir=str(tmp_path))
        runner.run(log_every=100)
        csv=[p for p in tmp_path.iterdir() if p.suffix=='.csv' and p not in paths][-1]
        paths.append(str(csv))
    out_dir=tmp_path / 'compare'
    compare_runs(paths, str(out_dir), metrics=('reward','cost'))
    for f in ['reward_compare.png','cost_compare.png']:
        assert (out_dir / f).exists(), f'Missing {f}'
