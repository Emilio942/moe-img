import torch
from rl.rl_agent import ReinforceAgent

def test_reinforce_agent_training():
    """Tests if the REINFORCE agent can learn a simple task."""
    input_dim = 4
    num_actions = 2
    agent = ReinforceAgent(input_dim, num_actions)

    # Dummy environment: state is a random vector, reward is 1 if action is 0, else 0
    for _ in range(100):
        state = torch.randn(input_dim)
        action = agent.select_action(state)
        reward = 1 if action == 0 else 0
        agent.rewards.append(reward)
        agent.finish_episode()

    # After training, the agent should have learned to prefer action 0
    state = torch.randn(input_dim)
    action_probs = agent.policy(state.unsqueeze(0))
    assert action_probs[0][0] > 0.7
