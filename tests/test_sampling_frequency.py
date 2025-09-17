import math, torch
from rl.rl_env import MoEGatingEnv
from rl.rl_agent import ReinforceAgent
from models.moe import Expert
from rl.rl_runner import RLRunner

class Dummy(torch.nn.Module):
    def __init__(self,n=3,fd=6,nc=4):
        super().__init__(); self.experts=torch.nn.ModuleList([Expert(fd,nc) for _ in range(n)])
    def forward(self,x):
        outs=[e(x) for e in self.experts]; return torch.stack(outs,dim=1).mean(dim=1), None

def test_probe_sampling_frequency(tmp_path):
    torch.manual_seed(0)
    model=Dummy()
    # sample_every=3 should reduce probe calls
    env=MoEGatingEnv(model,3,6,4, top_k=2, batch_size=4, sample_every=3)
    agent=ReinforceAgent(env.observation_dim,3)
    episodes=10
    runner=RLRunner(env, agent, episodes=episodes, out_dir=str(tmp_path))
    runner.run(log_every=100)
    # Expect roughly ceil(episodes/3) * num_experts probe calls (since each episode we probe each expert when measuring)
    # measurement occurs when episode_counter % sample_every == 0 (i.e., on episodes 3,6,9) => floor(episodes/sample_every)
    expected_calls=(episodes//3)*len(model.experts)
    assert env.probe_calls == expected_calls, f"probe_calls {env.probe_calls} != expected {expected_calls}"
