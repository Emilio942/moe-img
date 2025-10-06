import torch
import torch.nn as nn
import pytest
from optim.meta_optimizer import (
    MetaOptimizer, 
    OptimizerWrapper, 
    PlateauMetaOptimizer,
    ExpertGateMetaOptimizer,
    AdvancedPlateauMetaOptimizer,
    SmartExpertGateMetaOptimizer,
    RLMetaOptimizer
)


class DummyMetaOptimizer(MetaOptimizer):
    """Simple test meta-optimizer for unit testing."""
    
    def __init__(self):
        super().__init__(update_frequency=2)
        self.update_calls = []
    
    def update_hparams(self, metrics, step):
        self.update_calls.append((metrics, step))
        return {'lr': 0.001 * 0.9}  # Always reduce LR by 10%


def test_meta_optimizer_abstract_interface():
    """Test that MetaOptimizer enforces abstract interface."""
    # Cannot instantiate abstract class directly
    with pytest.raises(TypeError):
        MetaOptimizer()


def test_optimizer_wrapper_basic_functionality():
    """Test OptimizerWrapper basic pass-through functionality."""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Test without meta-optimizer
    wrapper = OptimizerWrapper(optimizer)
    
    # Basic operations should work
    wrapper.zero_grad()
    loss = torch.tensor(1.0, requires_grad=True)
    loss.backward()
    wrapper.step()
    
    # Properties should pass through
    assert len(wrapper.param_groups) == 1
    assert wrapper.param_groups[0]['lr'] == 0.1


def test_optimizer_wrapper_with_meta_optimizer():
    """Test OptimizerWrapper integration with meta-optimizer."""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    meta_opt = DummyMetaOptimizer()
    wrapper = OptimizerWrapper(optimizer, meta_opt)
    
    # First update - step=0, should trigger meta-optimizer (0 % 2 == 0)
    metrics = {'loss': 1.0, 'lr': 0.1}
    wrapper.update_hyperparameters(metrics)
    wrapper.step()
    assert len(meta_opt.update_calls) == 1
    
    # Second update - step=1, should not trigger meta-optimizer (1 % 2 != 0)
    wrapper.update_hyperparameters(metrics)
    wrapper.step()
    assert len(meta_opt.update_calls) == 1  # Still 1, no new call
    
    # Third update - step=2, should trigger meta-optimizer again (2 % 2 == 0)
    wrapper.update_hyperparameters(metrics)
    wrapper.step()
    assert len(meta_opt.update_calls) == 2
    
    # Learning rate should be updated (first update applied 0.001 * 0.9)
    assert wrapper.param_groups[0]['lr'] == 0.001 * 0.9


def test_advanced_plateau_meta_optimizer():
    """Test AdvancedPlateauMetaOptimizer with warmup and enhanced detection."""
    meta_opt = AdvancedPlateauMetaOptimizer(
        patience=2, 
        lr_factor=0.8, 
        warmup_steps=3,
        trend_window=3,
        grad_norm_threshold=5.0
    )
    
    # Test warmup phase
    for step in range(3):
        metrics = {'loss': 1.0, 'lr': 1e-3, 'grad_norm': 1.0}
        hparams = meta_opt.update_hparams(metrics, step)
        expected_lr = 1e-3 * (step + 1) / 3
        assert abs(hparams['lr'] - expected_lr) < 1e-6
    
    # Test plateau detection after warmup
    # Good loss initially
    metrics = {'loss': 0.5, 'lr': 1e-3, 'grad_norm': 1.0}
    hparams = meta_opt.update_hparams(metrics, 3)
    assert 'lr' not in hparams  # No change yet
    
    # Plateau - loss not improving
    for step in range(4, 7):
        metrics = {'loss': 0.6, 'lr': 1e-3, 'grad_norm': 1.0}
        hparams = meta_opt.update_hparams(metrics, step)
    
    # Should trigger LR reduction after patience
    assert 'lr' in hparams
    assert hparams['lr'] == 1e-3 * 0.8
    
    # Test weight decay adjustment with high gradient norm
    metrics = {'loss': 0.6, 'lr': 1e-3, 'grad_norm': 8.0, 'weight_decay': 1e-4}
    hparams = meta_opt.update_hparams(metrics, 7)
    assert 'weight_decay' in hparams
    assert hparams['weight_decay'] > 1e-4


def test_smart_expert_gate_meta_optimizer():
    """Test SmartExpertGateMetaOptimizer with sparsity and specialization."""
    meta_opt = SmartExpertGateMetaOptimizer(
        gate_lr_factor=0.1,
        expert_lr_factor=1.0,
        entropy_threshold=0.8,
        sparsity_target=0.7,
        specialization_bonus=1.2
    )
    
    # Test basic expert/gate LR separation
    metrics = {
        'lr': 1e-3,
        'gate_entropy': 1.0,  # High entropy
        'gate_sparsity': 0.7,  # At target
        'expert_usage': [0.3, 0.6, 0.1]  # Good specialization
    }
    hparams = meta_opt.update_hparams(metrics, 1)
    
    assert 'gate_lr' in hparams
    assert 'expert_lr' in hparams
    assert hparams['gate_lr'] == 1e-3 * 0.1  # Base gate LR
    assert hparams['expert_lr'] == 1e-3 * 1.0  # Base expert LR
    
    # Test low entropy adjustment
    metrics['gate_entropy'] = 0.5  # Low entropy
    hparams = meta_opt.update_hparams(metrics, 2)
    assert hparams['gate_lr'] < 1e-3 * 0.1  # Should be reduced
    
    # Test sparsity adjustment - too sparse
    for i in range(3):
        metrics['gate_sparsity'] = 0.9  # Too sparse
        meta_opt.update_hparams(metrics, i + 3)
    
    hparams = meta_opt.update_hparams(metrics, 6)
    # Should increase gate LR to encourage more exploration
    assert hparams['gate_lr'] > 1e-3 * 0.1 * 0.5  # Increased from low entropy baseline
    
    # Test specialization bonus
    metrics['expert_usage'] = [0.8, 0.1, 0.1]  # High variance = good specialization
    hparams = meta_opt.update_hparams(metrics, 7)
    assert hparams['expert_lr'] > 1e-3 * 1.0  # Should get bonus


def test_advanced_meta_optimizer_state_management():
    """Test state dict save/load for advanced meta-optimizers."""
    # Test AdvancedPlateauMetaOptimizer
    meta_opt = AdvancedPlateauMetaOptimizer(patience=2)
    
    # Build up some state
    for i in range(5):
        metrics = {'loss': 1.0 - i * 0.1, 'lr': 1e-3, 'grad_norm': 2.0}
        meta_opt.update_hparams(metrics, i)
    
    # Save state
    state_dict = meta_opt.state_dict()
    assert 'loss_history' in state_dict
    assert 'grad_norm_history' in state_dict
    assert len(state_dict['loss_history']) > 0
    
    # Create new optimizer and load state
    new_meta_opt = AdvancedPlateauMetaOptimizer(patience=2)
    new_meta_opt.load_state_dict(state_dict)
    
    assert new_meta_opt._loss_history == meta_opt._loss_history
    assert new_meta_opt._best_loss == meta_opt._best_loss
    
    # Test SmartExpertGateMetaOptimizer
    smart_meta_opt = SmartExpertGateMetaOptimizer()
    
    for i in range(3):
        metrics = {'lr': 1e-3, 'gate_entropy': 0.8, 'gate_sparsity': 0.7}
        smart_meta_opt.update_hparams(metrics, i)
    
    smart_state = smart_meta_opt.state_dict()
    assert 'entropy_history' in smart_state
    assert 'sparsity_history' in smart_state
    
    new_smart_meta_opt = SmartExpertGateMetaOptimizer()
    new_smart_meta_opt.load_state_dict(smart_state)
    
    assert new_smart_meta_opt._entropy_history == smart_meta_opt._entropy_history


def test_meta_optimizer_reset_advanced():
    """Test reset functionality for advanced meta-optimizers."""
    # Test AdvancedPlateauMetaOptimizer reset
    meta_opt = AdvancedPlateauMetaOptimizer()
    
    # Build up state
    for i in range(3):
        metrics = {'loss': 1.0, 'lr': 1e-3, 'grad_norm': 5.0, 'weight_norm': 2.0}
        meta_opt.update_hparams(metrics, i)
    
    assert len(meta_opt._loss_history) > 0
    assert len(meta_opt._grad_norm_history) > 0
    assert len(meta_opt._weight_norm_history) > 0
    
    meta_opt.reset()
    
    assert len(meta_opt._loss_history) == 0
    assert len(meta_opt._grad_norm_history) == 0
    assert len(meta_opt._weight_norm_history) == 0
    assert meta_opt._best_loss == float('inf')
    
    # Test SmartExpertGateMetaOptimizer reset
    smart_meta_opt = SmartExpertGateMetaOptimizer()
    
    for i in range(3):
        metrics = {'lr': 1e-3, 'gate_entropy': 0.8, 'gate_sparsity': 0.7}
        smart_meta_opt.update_hparams(metrics, i)
    
    assert len(smart_meta_opt._entropy_history) > 0
    
    smart_meta_opt.reset()
    
    assert len(smart_meta_opt._entropy_history) == 0
    assert len(smart_meta_opt._sparsity_history) == 0


def test_rl_meta_optimizer_basic():
    """Test basic RLMetaOptimizer functionality."""
    rl_meta_opt = RLMetaOptimizer(
        update_frequency=1,
        state_dim=8,
        hidden_dim=16,
        lr_actions=[0.8, 1.0, 1.2],
        wd_actions=[-1e-5, 0.0, 1e-5]
    )
    
    # Test action space setup
    assert rl_meta_opt.num_lr_actions == 3
    assert rl_meta_opt.num_wd_actions == 3
    assert rl_meta_opt.total_actions == 9
    
    # Test state computation
    metrics = {
        'loss': 1.5,
        'lr': 1e-3,
        'grad_norm': 5.0,
        'weight_decay': 1e-4,
        'cost': 0.3
    }
    
    state = rl_meta_opt._compute_state(metrics, 0)
    assert state.shape[0] == 8  # state_dim
    assert all(s >= -1.0 and s <= 1.0 for s in state.tolist())  # Normalized values
    
    # Test action selection
    action_idx, log_prob, lr_mult, wd_delta = rl_meta_opt._select_action(state)
    assert 0 <= action_idx < 9
    assert lr_mult in [0.8, 1.0, 1.2]
    assert wd_delta in [-1e-5, 0.0, 1e-5]
    assert log_prob <= 0  # Log probability should be negative


def test_rl_meta_optimizer_training():
    """Test RL meta-optimizer training with episode collection."""
    rl_meta_opt = RLMetaOptimizer(
        update_frequency=1,
        state_dim=8,
        hidden_dim=16,
        lr_actions=[0.9, 1.0, 1.1],
        wd_actions=[0.0]  # Single action to simplify
    )
    
    # Simulate training episodes with improving loss
    metrics_sequence = [
        {'loss': 2.0, 'lr': 1e-3, 'grad_norm': 3.0, 'weight_decay': 1e-4, 'cost': 0.5},
        {'loss': 1.8, 'lr': 1e-3, 'grad_norm': 2.8, 'weight_decay': 1e-4, 'cost': 0.4},
        {'loss': 1.6, 'lr': 1e-3, 'grad_norm': 2.5, 'weight_decay': 1e-4, 'cost': 0.35},
        {'loss': 1.4, 'lr': 1e-3, 'grad_norm': 2.2, 'weight_decay': 1e-4, 'cost': 0.3},
        {'loss': 1.2, 'lr': 1e-3, 'grad_norm': 2.0, 'weight_decay': 1e-4, 'cost': 0.25},
        {'loss': 1.0, 'lr': 1e-3, 'grad_norm': 1.8, 'weight_decay': 1e-4, 'cost': 0.2},
    ]
    
    # Run through episodes
    for step, metrics in enumerate(metrics_sequence):
        hparams = rl_meta_opt.update_hparams(metrics, step)
        # Should return hyperparameter updates
        assert isinstance(hparams, dict)
    
    # Should have collected episode data
    assert len(rl_meta_opt._episode_rewards) > 0
    assert len(rl_meta_opt._episode_states) > 0
    assert len(rl_meta_opt._episode_actions) > 0
    
    # After enough episodes, should trigger training
    # This will clear episode data
    initial_episode_count = len(rl_meta_opt._episode_rewards)
    if initial_episode_count >= 5:
        # Training should have been triggered and data cleared
        pass  # Data might be cleared by training
    
    print(f"RL meta-optimizer processed {len(metrics_sequence)} steps")


def test_rl_meta_optimizer_reward_computation():
    """Test reward computation for RL meta-optimizer."""
    rl_meta_opt = RLMetaOptimizer()
    
    # Test reward with no previous metrics
    reward = rl_meta_opt._compute_reward(None, {'loss': 1.0, 'cost': 0.5})
    assert reward == 0.0
    
    # Test positive reward (loss decreases)
    prev_metrics = {'loss': 2.0, 'cost': 0.6}
    curr_metrics = {'loss': 1.5, 'cost': 0.5}
    reward = rl_meta_opt._compute_reward(prev_metrics, curr_metrics)
    assert reward > 0  # Loss decreased by 0.5, cost decreased by 0.1
    
    # Test negative reward (loss increases)
    prev_metrics = {'loss': 1.0, 'cost': 0.3}
    curr_metrics = {'loss': 1.5, 'cost': 0.4}
    reward = rl_meta_opt._compute_reward(prev_metrics, curr_metrics)
    assert reward < 0  # Both loss and cost increased


def test_rl_meta_optimizer_state_management():
    """Test RL meta-optimizer state saving and loading."""
    rl_meta_opt = RLMetaOptimizer(hidden_dim=16)
    
    # Run a few updates to build state
    for i in range(3):
        metrics = {'loss': 2.0 - i * 0.2, 'lr': 1e-3, 'grad_norm': 3.0}
        rl_meta_opt.update_hparams(metrics, i)
    
    # Save state
    state_dict = rl_meta_opt.state_dict()
    assert 'policy_net' in state_dict
    assert 'policy_optimizer' in state_dict
    assert 'reward_baseline' in state_dict
    
    # Create new optimizer and load state
    new_rl_meta_opt = RLMetaOptimizer(hidden_dim=16)
    new_rl_meta_opt.load_state_dict(state_dict)
    
    # States should match
    assert new_rl_meta_opt._reward_baseline == rl_meta_opt._reward_baseline
    assert len(new_rl_meta_opt._state_history) == len(rl_meta_opt._state_history)
    
    # Policy networks should have same parameters
    original_params = list(rl_meta_opt.policy_net.parameters())
    loaded_params = list(new_rl_meta_opt.policy_net.parameters())
    
    for orig, loaded in zip(original_params, loaded_params):
        assert torch.allclose(orig, loaded, atol=1e-6)


def test_rl_meta_optimizer_reset():
    """Test RL meta-optimizer reset functionality."""
    rl_meta_opt = RLMetaOptimizer()
    
    # Build up some state
    for i in range(4):
        metrics = {'loss': 2.0 - i * 0.1, 'lr': 1e-3, 'grad_norm': 2.0}
        rl_meta_opt.update_hparams(metrics, i)
    
    # Should have collected some state
    assert len(rl_meta_opt._state_history) > 0
    assert len(rl_meta_opt._episode_rewards) > 0
    
    # Reset and verify cleanup
    rl_meta_opt.reset()
    
    assert len(rl_meta_opt._state_history) == 0
    assert len(rl_meta_opt._episode_rewards) == 0
    assert len(rl_meta_opt._episode_states) == 0
    assert rl_meta_opt._reward_baseline == 0.0
    assert not hasattr(rl_meta_opt, '_prev_metrics')


def test_plateau_meta_optimizer():
    """Test PlateauMetaOptimizer plateau detection and LR reduction."""
    meta_opt = PlateauMetaOptimizer(patience=2, lr_factor=0.5, min_lr=1e-5)
    
    # Simulate improving loss - should not reduce LR
    metrics1 = {'loss': 1.0, 'lr': 1e-3}
    hparams1 = meta_opt.update_hparams(metrics1, 1)
    assert 'lr' not in hparams1  # No LR change
    
    metrics2 = {'loss': 0.8, 'lr': 1e-3}
    hparams2 = meta_opt.update_hparams(metrics2, 2)
    assert 'lr' not in hparams2  # Still improving
    
    # Simulate plateau - loss stops improving
    metrics3 = {'loss': 0.9, 'lr': 1e-3}  # Worse than previous
    hparams3 = meta_opt.update_hparams(metrics3, 3)
    assert 'lr' not in hparams3  # First plateau step
    
    metrics4 = {'loss': 0.85, 'lr': 1e-3}  # Still not better than best (0.8)
    hparams4 = meta_opt.update_hparams(metrics4, 4)
    assert 'lr' in hparams4  # Should reduce LR after patience=2
    assert hparams4['lr'] == 1e-3 * 0.5


def test_plateau_meta_optimizer_weight_decay():
    """Test PlateauMetaOptimizer weight decay adjustment."""
    meta_opt = PlateauMetaOptimizer()
    
    # High gradient norm should increase weight decay
    metrics = {
        'loss': 1.0,
        'grad_norm': 15.0,  # Above threshold of 10.0
        'weight_decay': 1e-4
    }
    hparams = meta_opt.update_hparams(metrics, 1)
    assert 'weight_decay' in hparams
    assert hparams['weight_decay'] > 1e-4


def test_expert_gate_meta_optimizer():
    """Test ExpertGateMetaOptimizer expert/gate differentiation."""
    meta_opt = ExpertGateMetaOptimizer(
        gate_lr_factor=0.1,
        expert_lr_factor=1.0,
        entropy_threshold=0.8
    )
    
    # High entropy (exploring) case
    metrics_high_entropy = {
        'lr': 1e-3,
        'gate_entropy': 1.2  # Above threshold
    }
    hparams_high = meta_opt.update_hparams(metrics_high_entropy, 1)
    assert hparams_high['gate_lr'] == 1e-3 * 0.1  # Normal gate LR
    assert hparams_high['expert_lr'] == 1e-3 * 1.0
    
    # Low entropy (confident) case
    metrics_low_entropy = {
        'lr': 1e-3,
        'gate_entropy': 0.5  # Below threshold
    }
    hparams_low = meta_opt.update_hparams(metrics_low_entropy, 2)
    assert hparams_low['gate_lr'] == 1e-3 * 0.1 * 0.5  # Reduced gate LR
    assert hparams_low['expert_lr'] == 1e-3 * 1.0


def test_meta_optimizer_reset():
    """Test meta-optimizer state reset functionality."""
    meta_opt = PlateauMetaOptimizer(patience=2)
    
    # Build up some state
    meta_opt.update_hparams({'loss': 1.0, 'lr': 1e-3}, 1)
    meta_opt.update_hparams({'loss': 1.1, 'lr': 1e-3}, 2)
    
    assert meta_opt._patience_counter > 0
    assert len(meta_opt._loss_history) > 0
    
    # Reset should clear state
    meta_opt.reset()
    assert meta_opt._patience_counter == 0
    assert len(meta_opt._loss_history) == 0
    assert meta_opt._best_loss == float('inf')


def test_optimizer_wrapper_state_dict():
    """Test state dict save/load for OptimizerWrapper."""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    wrapper = OptimizerWrapper(optimizer)
    
    # Take a step to create state
    loss = torch.tensor(1.0, requires_grad=True)
    wrapper.zero_grad()
    loss.backward()
    wrapper.step()
    
    # Save and load state
    state_dict = wrapper.state_dict()
    
    new_optimizer = torch.optim.AdamW(model.parameters(), lr=0.2)  # Different LR
    new_wrapper = OptimizerWrapper(new_optimizer)
    new_wrapper.load_state_dict(state_dict)
    
    # State should be preserved
    assert new_wrapper.param_groups[0]['lr'] == 0.1  # Original LR restored