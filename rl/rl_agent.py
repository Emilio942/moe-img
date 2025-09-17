import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class Policy(nn.Module):
    """A simple MLP policy network."""
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        logits = self.layer2(x)
        # Stabilize softmax by subtracting max for numerical stability
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.softmax(logits, dim=-1)
        # Clamp to avoid exact 0 which can cause inf log probs in downstream RL math
        probs = probs.clamp_min(1e-8)
        return probs

class ReinforceAgent:
    """A REINFORCE agent for learning the gating policy."""
    def __init__(self, input_dim, num_actions, learning_rate: float = 1e-3):
        self.policy = Policy(input_dim, num_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.saved_log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.entropies: list[float] = []  # per action entropy

    def select_action(self, state: torch.Tensor) -> int:
        state = state.unsqueeze(0)
        probs = self.policy(state)
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones_like(probs) / probs.shape[-1]
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        with torch.no_grad():
            self.entropies.append(float(m.entropy().item()))
        return int(action.item())

    def finish_episode(self) -> None:
        R = 0.0
        returns: list[float] = []
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        if not returns:
            return
        returns_t = torch.tensor(returns, dtype=torch.float32)
        if returns_t.numel() > 1:
            std = returns_t.std(unbiased=False)
            if std > 1e-9:
                returns_t = (returns_t - returns_t.mean()) / (std + 1e-9)
        policy_loss = []
        for log_prob, ret in zip(self.saved_log_probs, returns_t):
            policy_loss.append(-log_prob * ret)
        if not policy_loss:
            return
        loss = torch.stack(policy_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.rewards.clear()
        self.saved_log_probs.clear()
        # keep entropies for analysis
