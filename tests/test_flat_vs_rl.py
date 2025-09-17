import torch, json
from rl.rl_env import MoEGatingEnv
from rl.rl_agent import ReinforceAgent
from rl.rl_runner import RLRunner
from rl.rl_baseline_runner import FlatBaselineRunner
from rl.rl_compare import compare
from models.moe import Expert

class Dummy(torch.nn.Module):
    def __init__(self, n=3, fd=6, nc=4):
        super().__init__(); self.experts=torch.nn.ModuleList([Expert(fd,nc) for _ in range(n)])
    def forward(self,x):
        outs=[e(x) for e in self.experts]; return torch.stack(outs,dim=1).mean(dim=1), None

def test_flat_vs_rl_comparison(tmp_path):
    torch.manual_seed(0)
    model=Dummy()
    env=MoEGatingEnv(model,3,6,4, top_k=2, batch_size=4)
    # Baseline run
    flat=FlatBaselineRunner(env, episodes=6, out_dir=str(tmp_path), seed=0)
    flat_rows=flat.run(log_every=100)
    # RL run
    agent=ReinforceAgent(env.observation_dim,3)
    rl=RLRunner(env, agent, episodes=6, out_dir=str(tmp_path))
    rl_rows=rl.run(log_every=100)
    flat_csv=[p for p in tmp_path.iterdir() if p.suffix=='.csv' and p.stem.startswith('flat_')][0]
    rl_csv=[p for p in tmp_path.iterdir() if p.suffix=='.csv' and not p.stem.startswith('flat_')][-1]
    out_json=tmp_path / 'comparison' / 'flat_vs_rl.json'
    out_md=tmp_path / 'comparison' / 'flat_vs_rl.md'
    result=compare(str(flat_csv), str(rl_csv), str(out_json), str(out_md))
    assert out_json.exists() and out_md.exists()
    # Basic sanity: diff keys present
    assert 'diff' in result and 'mean_accuracy' in result['diff']
