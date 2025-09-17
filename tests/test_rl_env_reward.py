import torch
from rl.rl_env import MoEGatingEnv
from models.moe import VectorMoE

def test_rl_env_reward_basic():
    input_dim = 32
    num_classes = 10
    num_experts = 4
    top_k = 2
    model = VectorMoE(input_dim, num_classes, num_experts, top_k)
    env = MoEGatingEnv(model=model, num_experts=num_experts, feature_dim=input_dim, num_classes=num_classes, top_k=top_k, batch_size=8)

    obs, info = env.reset()
    assert obs.shape[0] == env.observation_dim

    action = [0, 1]
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert terminated is True
    assert isinstance(reward, float)
    assert 'accuracy' in info and 'cost' in info
    # Reward should not be exactly zero in normal case
    assert reward != 0.0
    assert len(info['action']) == top_k


def test_rl_env_action_dedup():
    input_dim = 16
    num_classes = 5
    num_experts = 3
    top_k = 2
    model = VectorMoE(input_dim, num_classes, num_experts, top_k)
    env = MoEGatingEnv(model=model, num_experts=num_experts, feature_dim=input_dim, num_classes=num_classes, top_k=top_k, batch_size=4)

    env.reset()
    # Provide duplicate action indices; env should deduplicate and pad
    action = [1, 1]
    _, _, _, _, info = env.step(action)
    assert len(set(info['action'])) == top_k
