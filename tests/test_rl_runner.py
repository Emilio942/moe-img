import os
import csv
import json
import torch
from rl.rl_runner import RLRunner
from rl.rl_env import MoEGatingEnv
from rl.rl_agent import ReinforceAgent
from models.moe import Expert

class DummyModel(torch.nn.Module):
    def __init__(self, n, fd, nc):
        super().__init__(); self.experts=torch.nn.ModuleList([Expert(fd,nc) for _ in range(n)])
    def forward(self,x):
        outs=[e(x) for e in self.experts]
        return torch.stack(outs,dim=1).mean(dim=1), None

def test_rl_runner_creates_outputs(tmp_path):
    torch.manual_seed(0)
    num_experts=3; feature_dim=6; num_classes=4
    model=DummyModel(num_experts, feature_dim, num_classes)
    env=MoEGatingEnv(model, num_experts, feature_dim, num_classes, top_k=2, batch_size=4)
    agent=ReinforceAgent(input_dim=env.observation_dim, num_actions=num_experts)
    runner=RLRunner(env, agent, episodes=5, out_dir=str(tmp_path))
    rows=runner.run(log_every=10)
    assert len(rows)==5
    # CSV exists
    csv_files=[p for p in tmp_path.iterdir() if p.suffix=='.csv']
    assert csv_files, 'CSV not created'
    csv_path=csv_files[0]
    with csv_path.open() as f:
        reader=csv.DictReader(f)
        headers=reader.fieldnames
        assert headers==['episode','reward','accuracy','cost','action','entropy','ema_reward','ema_entropy','moving_avg_reward']
        data=list(reader)
        assert len(data)==5
        ent_vals=[float(row['entropy']) for row in data if row['entropy'] not in (None,'')]
        assert all(v >= 0.0 for v in ent_vals)
    # Meta
    meta_files=[p for p in tmp_path.iterdir() if p.name.endswith('_meta.json')]
    assert meta_files, 'Meta file missing'
    meta=json.loads(meta_files[0].read_text())
    assert 'pareto' in meta and len(meta['pareto'])>=1
