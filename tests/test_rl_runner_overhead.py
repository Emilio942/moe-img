import json
from pathlib import Path
import torch
from rl.rl_runner import RLRunner
from rl.rl_agent import ReinforceAgent
from rl.rl_env import MoEGatingEnv
from models.moe import Expert


class Dummy(torch.nn.Module):
    def __init__(self, n=3, fd=6, nc=4):
        super().__init__()
        self.experts = torch.nn.ModuleList([Expert(fd, nc) for _ in range(n)])
    def forward(self, x):
        outs = [e(x) for e in self.experts]
        return torch.stack(outs, dim=1).mean(dim=1), None


def test_rlrunner_writes_monitor_log_and_meta(tmp_path):
    torch.manual_seed(0)
    model = Dummy()
    env = MoEGatingEnv(model, 3, 6, 4, top_k=2, batch_size=4)
    agent = ReinforceAgent(env.observation_dim, 3)
    log_path = tmp_path / 'mon.jsonl'
    runner = RLRunner(env, agent, episodes=3, out_dir=str(tmp_path), monitor_log_path=str(log_path))
    rows = runner.run(log_every=100)
    assert len(rows) == 3
    # Monitor log exists with 3 lines
    assert log_path.exists()
    lines = [l for l in log_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    rec = json.loads(lines[0])
    assert 'epoch' in rec and 'avg_time_ms' in rec
