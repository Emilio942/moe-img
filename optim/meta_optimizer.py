from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MetaOptimizer(ABC):
    """Abstract base class for meta-optimizers that adjust hyperparameters during training.
    
    A meta-optimizer observes training metrics and dynamically adjusts the hyperparameters
    of the main optimizer (learning rate, weight decay, etc.) to improve training stability
    and convergence.
    """
    
    def __init__(self, update_frequency: int = 1):
        """
        Args:
            update_frequency: How often to update hyperparameters (every N steps/epochs)
        """
        self.update_frequency = update_frequency
        self._step_count = 0
    
    @abstractmethod
    def update_hparams(self, metrics: Dict[str, float], step: int) -> Dict[str, Any]:
        """Update hyperparameters based on current metrics.
        
        Args:
            metrics: Dictionary containing training metrics like loss, accuracy, grad_norm, etc.
            step: Current training step/epoch
            
        Returns:
            Dictionary of new hyperparameter values to apply
        """
        pass
    
    def should_update(self, step: int) -> bool:
        """Check if hyperparameters should be updated at this step."""
        return step % self.update_frequency == 0
    
    def reset(self):
        """Reset internal state of the meta-optimizer."""
        self._step_count = 0


class OptimizerWrapper:
    """Wrapper around PyTorch optimizers that integrates with MetaOptimizer.
    
    This wrapper intercepts optimizer calls and periodically updates hyperparameters
    using a MetaOptimizer instance.
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer, 
                 meta_optimizer: Optional[MetaOptimizer] = None):
        """
        Args:
            optimizer: The main PyTorch optimizer to wrap
            meta_optimizer: Meta-optimizer instance for hyperparameter updates
        """
        self.optimizer = optimizer
        self.meta_optimizer = meta_optimizer
        self._step_count = 0
        self._metrics_history = []
        
    def step(self, closure=None):
        """Perform optimization step, potentially updating hyperparameters."""
        # Perform the actual optimization step
        loss = self.optimizer.step(closure)
        self._step_count += 1
        return loss
    
    def update_hyperparameters(self, metrics: Dict[str, float]):
        """Update hyperparameters using the meta-optimizer.
        
        Args:
            metrics: Current training metrics (loss, accuracy, etc.)
        """
        if self.meta_optimizer is None:
            return
            
        if self.meta_optimizer.should_update(self._step_count):
            new_hparams = self.meta_optimizer.update_hparams(metrics, self._step_count)
            self._apply_hyperparameters(new_hparams)
    
    def _apply_hyperparameters(self, hparams: Dict[str, Any]):
        """Apply new hyperparameters to the optimizer."""
        for param_group in self.optimizer.param_groups:
            for key, value in hparams.items():
                if key in param_group:
                    param_group[key] = value
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients - pass through to underlying optimizer."""
        return self.optimizer.zero_grad(set_to_none)
    
    def state_dict(self):
        """Get state dict - pass through to underlying optimizer."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict - pass through to underlying optimizer."""
        return self.optimizer.load_state_dict(state_dict)
    
    @property
    def param_groups(self):
        """Access parameter groups - pass through to underlying optimizer."""
        return self.optimizer.param_groups


class PlateauMetaOptimizer(MetaOptimizer):
    """Simple heuristic meta-optimizer that reduces learning rate on loss plateaus.
    
    This implements a basic strategy:
    - Reduce LR by a factor when validation loss plateaus for N epochs
    - Increase weight decay when gradient norms suggest overfitting
    """
    
    def __init__(self, 
                 update_frequency: int = 1,
                 patience: int = 3,
                 lr_factor: float = 0.5,
                 min_lr: float = 1e-6,
                 wd_increase_factor: float = 1.1):
        """
        Args:
            update_frequency: How often to check for updates
            patience: Number of epochs to wait before reducing LR
            lr_factor: Factor to multiply LR by when reducing
            min_lr: Minimum learning rate
            wd_increase_factor: Factor to increase weight decay by
        """
        super().__init__(update_frequency)
        self.patience = patience
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.wd_increase_factor = wd_increase_factor
        
        self._best_loss = float('inf')
        self._patience_counter = 0
        self._loss_history = []
    
    def update_hparams(self, metrics: Dict[str, float], step: int) -> Dict[str, Any]:
        """Update hyperparameters based on loss plateau detection."""
        current_loss = metrics.get('loss', float('inf'))
        self._loss_history.append(current_loss)
        
        new_hparams = {}
        
        # Learning rate reduction on plateau
        if current_loss < self._best_loss:
            self._best_loss = current_loss
            self._patience_counter = 0
        else:
            self._patience_counter += 1
            
        if self._patience_counter >= self.patience:
            # Reduce learning rate
            current_lr = metrics.get('lr', 1e-3)
            new_lr = max(current_lr * self.lr_factor, self.min_lr)
            new_hparams['lr'] = new_lr
            self._patience_counter = 0  # Reset counter
            
        # Weight decay adjustment based on gradient norm
        grad_norm = metrics.get('grad_norm', 0.0)
        if grad_norm > 10.0:  # High gradient norm suggests potential overfitting
            current_wd = metrics.get('weight_decay', 0.0)
            new_hparams['weight_decay'] = current_wd * self.wd_increase_factor
            
        return new_hparams
    
    def reset(self):
        """Reset internal state."""
        super().reset()
        self._best_loss = float('inf')
        self._patience_counter = 0
        self._loss_history = []


class ExpertGateMetaOptimizer(MetaOptimizer):
    """Meta-optimizer that treats experts and gate parameters differently.
    
    This implements separate learning rates for:
    - Gate parameters (more conservative updates)
    - Expert parameters (more aggressive updates)
    """
    
    def __init__(self, 
                 update_frequency: int = 1,
                 gate_lr_factor: float = 0.1,
                 expert_lr_factor: float = 1.0,
                 entropy_threshold: float = 0.8):
        """
        Args:
            update_frequency: How often to update hyperparameters
            gate_lr_factor: Relative learning rate for gate parameters
            expert_lr_factor: Relative learning rate for expert parameters
            entropy_threshold: Gate entropy threshold for LR adjustment
        """
        super().__init__(update_frequency)
        self.gate_lr_factor = gate_lr_factor
        self.expert_lr_factor = expert_lr_factor
        self.entropy_threshold = entropy_threshold
    
    def update_hparams(self, metrics: Dict[str, float], step: int) -> Dict[str, Any]:
        """Update hyperparameters with expert/gate-specific logic."""
        base_lr = metrics.get('lr', 1e-3)
        gate_entropy = metrics.get('gate_entropy', 1.0)
        
        new_hparams = {}
        
        # Adjust gate learning rate based on entropy
        if gate_entropy < self.entropy_threshold:
            # Low entropy: gate is confident, reduce learning rate
            gate_lr = base_lr * self.gate_lr_factor * 0.5
        else:
            # High entropy: gate is exploring, normal learning rate
            gate_lr = base_lr * self.gate_lr_factor
            
        # Expert learning rate stays more aggressive
        expert_lr = base_lr * self.expert_lr_factor
        
        # Note: This assumes parameter groups are organized by component
        # In practice, this would need to be coordinated with the optimizer setup
        new_hparams['gate_lr'] = gate_lr
        new_hparams['expert_lr'] = expert_lr
        
        return new_hparams


class RLMetaOptimizer(MetaOptimizer):
    """RL-based meta-optimizer using REINFORCE to optimize hyperparameters.
    
    Uses a small MLP policy network to select hyperparameter adjustments based on
    training metrics (loss trends, gradient norms, cost metrics).
    """
    
    def __init__(self,
                 update_frequency: int = 10,
                 state_dim: int = 8,
                 hidden_dim: int = 32,
                 lr_actions: list = None,
                 wd_actions: list = None,
                 policy_lr: float = 1e-4,
                 baseline_decay: float = 0.9,
                 entropy_weight: float = 0.01,
                 reward_scale: float = 1.0):
        """
        Args:
            update_frequency: How often to update hyperparameters
            state_dim: Dimension of state representation
            hidden_dim: Hidden dimension of policy network
            lr_actions: List of LR multiplication factors [0.5, 1.0, 2.0]
            wd_actions: List of WD addition deltas [-1e-5, 0, 1e-5]
            policy_lr: Learning rate for policy network
            baseline_decay: Decay rate for baseline (moving average)
            entropy_weight: Weight for entropy regularization
            reward_scale: Scaling factor for rewards
        """
        super().__init__(update_frequency)
        
        # Action spaces
        self.lr_actions = lr_actions or [0.5, 1.0, 2.0]
        self.wd_actions = wd_actions or [-1e-5, 0.0, 1e-5]
        self.num_lr_actions = len(self.lr_actions)
        self.num_wd_actions = len(self.wd_actions)
        self.total_actions = self.num_lr_actions * self.num_wd_actions
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.total_actions)
        )
        
        # Optimizer for policy network
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        
        # Training parameters
        self.baseline_decay = baseline_decay
        self.entropy_weight = entropy_weight
        self.reward_scale = reward_scale
        
        # State tracking
        self._state_history = []
        self._reward_baseline = 0.0
        self._episode_rewards = []
        self._episode_states = []
        self._episode_actions = []
        self._episode_log_probs = []
        
    def _compute_state(self, metrics: dict, step: int) -> torch.Tensor:
        """Compute state representation from current metrics."""
        
        # Extract metrics with defaults
        current_loss = metrics.get('loss', 1.0)
        current_lr = metrics.get('lr', 1e-3)
        grad_norm = metrics.get('grad_norm', 1.0)
        weight_decay = metrics.get('weight_decay', 0.0)
        cost = metrics.get('cost', 0.0)
        
        # Track metrics for trend computation
        self._state_history.append({
            'loss': current_loss,
            'lr': current_lr,
            'grad_norm': grad_norm,
            'step': step
        })
        
        # Keep only recent history
        if len(self._state_history) > 20:
            self._state_history.pop(0)
        
        # Compute features
        features = []
        
        # Current metrics (normalized)
        features.append(min(current_loss / 10.0, 1.0))  # Loss (capped)
        features.append(min(current_lr * 1000, 1.0))     # LR (scaled)
        features.append(min(grad_norm / 50.0, 1.0))      # Grad norm (capped)
        features.append(min(weight_decay * 10000, 1.0))  # WD (scaled)
        features.append(min(cost, 1.0))                  # Cost (capped)
        
        # Trend features (if we have history)
        if len(self._state_history) >= 3:
            recent_losses = [h['loss'] for h in self._state_history[-3:]]
            loss_trend = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
            features.append(max(-1.0, min(loss_trend * 10, 1.0)))  # Loss trend
            
            recent_grads = [h['grad_norm'] for h in self._state_history[-3:]]
            grad_trend = (recent_grads[-1] - recent_grads[0]) / len(recent_grads)
            features.append(max(-1.0, min(grad_trend / 10, 1.0)))  # Grad trend
        else:
            features.extend([0.0, 0.0])  # No trend available
        
        # Step progress (normalized)
        features.append(min(step / 1000.0, 1.0))
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _compute_reward(self, prev_metrics: dict, curr_metrics: dict) -> float:
        """Compute reward based on improvement in loss and cost."""
        if prev_metrics is None:
            return 0.0
        
        prev_loss = prev_metrics.get('loss', 1.0)
        curr_loss = curr_metrics.get('loss', 1.0)
        prev_cost = prev_metrics.get('cost', 0.0)
        curr_cost = curr_metrics.get('cost', 0.0)
        
        # Loss improvement reward (primary objective)
        loss_improvement = prev_loss - curr_loss
        
        # Cost reduction reward (secondary objective)
        cost_improvement = prev_cost - curr_cost
        
        # Combined reward with loss being primary
        reward = loss_improvement + 0.1 * cost_improvement
        
        return reward * self.reward_scale
    
    def _select_action(self, state: torch.Tensor) -> tuple:
        """Select action using policy network."""
        
        with torch.no_grad():
            logits = self.policy_net(state)
            probs = F.softmax(logits, dim=-1)
            
            # Sample action
            action_idx = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[action_idx])
            
        # Convert flat action index to (lr_action, wd_action)
        lr_action_idx = action_idx // self.num_wd_actions
        wd_action_idx = action_idx % self.num_wd_actions
        
        lr_multiplier = self.lr_actions[lr_action_idx]
        wd_delta = self.wd_actions[wd_action_idx]
        
        return action_idx, log_prob, lr_multiplier, wd_delta
    
    def update_hparams(self, metrics: dict, step: int) -> dict:
        """RL-based hyperparameter update."""
        
        # Compute state
        state = self._compute_state(metrics, step)
        
        # Select action
        action_idx, log_prob, lr_mult, wd_delta = self._select_action(state)
        
        # Store episode data
        self._episode_states.append(state)
        self._episode_actions.append(action_idx)
        self._episode_log_probs.append(log_prob)
        
        # Compute reward if we have previous metrics
        prev_metrics = getattr(self, '_prev_metrics', None)
        reward = self._compute_reward(prev_metrics, metrics)
        self._episode_rewards.append(reward)
        
        # Store current metrics for next iteration
        self._prev_metrics = metrics.copy()
        
        # Apply actions to hyperparameters
        current_lr = metrics.get('lr', 1e-3)
        current_wd = metrics.get('weight_decay', 0.0)
        
        new_hparams = {}
        
        # Apply LR action (with bounds)
        new_lr = current_lr * lr_mult
        new_lr = max(1e-7, min(new_lr, 1.0))  # Clamp to reasonable range
        if abs(new_lr - current_lr) > 1e-8:
            new_hparams['lr'] = new_lr
        
        # Apply WD action (with bounds)
        new_wd = current_wd + wd_delta
        new_wd = max(0.0, min(new_wd, 1e-2))  # Clamp to reasonable range
        if abs(new_wd - current_wd) > 1e-8:
            new_hparams['weight_decay'] = new_wd
        
        # Train policy if we have enough experience
        if len(self._episode_rewards) >= 5:  # Mini-batch size
            self._train_policy()
            self._clear_episode()
        
        return new_hparams
    
    def _train_policy(self):
        """Train the policy network using REINFORCE."""
        
        if len(self._episode_rewards) == 0:
            return
        
        # Convert to tensors
        rewards = torch.tensor(self._episode_rewards, dtype=torch.float32)
        log_probs = torch.stack(self._episode_log_probs)
        
        # Compute returns (simple undiscounted for now)
        returns = rewards
        
        # Update baseline (moving average of returns)
        mean_return = returns.mean().item()
        self._reward_baseline = (self.baseline_decay * self._reward_baseline + 
                                (1 - self.baseline_decay) * mean_return)
        
        # Compute advantages
        advantages = returns - self._reward_baseline
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss (REINFORCE)
        policy_loss = -(log_probs * advantages).mean()
        
        # Entropy regularization
        states = torch.stack(self._episode_states)
        logits = self.policy_net(states)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -self.entropy_weight * entropy
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        # Update policy
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
    
    def _clear_episode(self):
        """Clear episode data."""
        self._episode_rewards.clear()
        self._episode_states.clear()
        self._episode_actions.clear()
        self._episode_log_probs.clear()
    
    def reset(self):
        """Reset internal state."""
        super().reset()
        self._state_history.clear()
        self._reward_baseline = 0.0
        self._clear_episode()
        if hasattr(self, '_prev_metrics'):
            delattr(self, '_prev_metrics')
    
    def state_dict(self):
        """Return state for checkpointing."""
        return {
            'policy_net': self.policy_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'reward_baseline': self._reward_baseline,
            'state_history': self._state_history,
            'step_count': self._step_count
        }
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.policy_net.load_state_dict(state_dict['policy_net'])
        self.policy_optimizer.load_state_dict(state_dict['policy_optimizer'])
        self._reward_baseline = state_dict.get('reward_baseline', 0.0)
        self._state_history = state_dict.get('state_history', [])
        self._step_count = state_dict.get('step_count', 0)


class AdvancedPlateauMetaOptimizer(MetaOptimizer):
    """Enhanced plateau-based meta-optimizer with multiple heuristic strategies.
    
    Features:
    - Multi-step plateau detection
    - Weight norm monitoring for overfitting detection  
    - Gradient norm-based weight decay adjustment
    - Loss trend analysis
    - Warmup and cooldown phases
    """
    
    def __init__(self, 
                 update_frequency: int = 1,
                 patience: int = 3,
                 lr_factor: float = 0.5,
                 min_lr: float = 1e-7,
                 grad_norm_threshold: float = 10.0,
                 weight_decay_factor: float = 1.5,
                 weight_norm_threshold: float = 1.0,
                 warmup_steps: int = 50,
                 trend_window: int = 5):
        """
        Args:
            update_frequency: How often to check for updates
            patience: Number of epochs to wait before reducing LR
            lr_factor: Factor to multiply LR by when reducing
            min_lr: Minimum learning rate
            grad_norm_threshold: Gradient norm above which to increase weight decay
            weight_decay_factor: Factor to multiply weight decay when increasing
            weight_norm_threshold: Weight norm growth rate threshold
            warmup_steps: Number of warmup steps with gradual LR increase
            trend_window: Window size for loss trend analysis
        """
        super().__init__(update_frequency)
        self.patience = patience
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.grad_norm_threshold = grad_norm_threshold
        self.weight_decay_factor = weight_decay_factor
        self.weight_norm_threshold = weight_norm_threshold
        self.warmup_steps = warmup_steps
        self.trend_window = trend_window
        
        # State tracking
        self._best_loss = float('inf')
        self._patience_counter = 0
        self._loss_history = []
        self._weight_norm_history = []
        self._grad_norm_history = []
        self._initial_lr = None
        
    def update_hparams(self, metrics: Dict[str, float], step: int) -> Dict[str, Any]:
        """Enhanced hyperparameter adjustment with multiple strategies."""
        current_loss = metrics.get('loss', float('inf'))
        current_lr = metrics.get('lr', 1e-3)
        grad_norm = metrics.get('grad_norm', 0.0)
        weight_decay = metrics.get('weight_decay', 0.0)
        weight_norm = metrics.get('weight_norm', 0.0)
        
        if self._initial_lr is None:
            self._initial_lr = current_lr
        
        # Track metrics
        self._loss_history.append(current_loss)
        self._grad_norm_history.append(grad_norm)
        if weight_norm > 0:
            self._weight_norm_history.append(weight_norm)
        
        # Limit history size
        for history in [self._loss_history, self._grad_norm_history, self._weight_norm_history]:
            while len(history) > max(self.trend_window, self.patience) + 2:
                history.pop(0)
        
        new_hparams = {}
        
        # 1. Warmup Phase - gradual LR increase
        if step < self.warmup_steps:
            warmup_factor = (step + 1) / self.warmup_steps
            new_hparams['lr'] = self._initial_lr * warmup_factor
            return new_hparams
        
        # 2. Plateau Detection with Enhanced Logic
        if current_loss < self._best_loss - 1e-6:  # Small threshold for numerical stability
            self._best_loss = current_loss
            self._patience_counter = 0
        else:
            self._patience_counter += 1
        
        # Multi-criteria plateau detection
        plateau_detected = False
        
        # Standard patience-based detection
        if self._patience_counter >= self.patience:
            plateau_detected = True
        
        # Trend-based detection (loss not improving over trend window)
        if len(self._loss_history) >= self.trend_window:
            recent_losses = self._loss_history[-self.trend_window:]
            if all(l >= recent_losses[0] * 0.999 for l in recent_losses[1:]):
                plateau_detected = True
        
        if plateau_detected and current_lr > self.min_lr:
            new_lr = max(current_lr * self.lr_factor, self.min_lr)
            new_hparams['lr'] = new_lr
            self._patience_counter = 0
        
        # 3. Weight Decay Adjustment
        should_increase_wd = False
        
        # High gradient norms suggest need for regularization
        if grad_norm > self.grad_norm_threshold:
            should_increase_wd = True
        
        # Growing weight norms suggest overfitting
        if len(self._weight_norm_history) >= 3:
            recent_norms = self._weight_norm_history[-3:]
            growth_rate = (recent_norms[-1] - recent_norms[0]) / len(recent_norms)
            if growth_rate > self.weight_norm_threshold:
                should_increase_wd = True
        
        # Loss increasing while gradients are high
        if len(self._loss_history) >= 2 and len(self._grad_norm_history) >= 2:
            loss_trend = self._loss_history[-1] - self._loss_history[-2]
            recent_grad_norm = sum(self._grad_norm_history[-2:]) / 2
            if loss_trend > 0 and recent_grad_norm > self.grad_norm_threshold * 0.5:
                should_increase_wd = True
        
        if should_increase_wd and weight_decay >= 0:
            new_hparams['weight_decay'] = max(weight_decay * self.weight_decay_factor, 1e-6)
        
        return new_hparams
    
    def reset(self):
        """Reset internal state."""
        super().reset()
        self._best_loss = float('inf')
        self._patience_counter = 0
        self._loss_history = []
        self._weight_norm_history = []
        self._grad_norm_history = []
        self._initial_lr = None
    
    def state_dict(self):
        """Return state for checkpointing."""
        return {
            'best_loss': self._best_loss,
            'patience_counter': self._patience_counter,
            'loss_history': self._loss_history,
            'weight_norm_history': self._weight_norm_history,
            'grad_norm_history': self._grad_norm_history,
            'initial_lr': self._initial_lr,
            'step_count': self._step_count
        }
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self._best_loss = state_dict.get('best_loss', float('inf'))
        self._patience_counter = state_dict.get('patience_counter', 0)
        self._loss_history = state_dict.get('loss_history', [])
        self._weight_norm_history = state_dict.get('weight_norm_history', [])
        self._grad_norm_history = state_dict.get('grad_norm_history', [])
        self._initial_lr = state_dict.get('initial_lr', None)
        self._step_count = state_dict.get('step_count', 0)


class SmartExpertGateMetaOptimizer(MetaOptimizer):
    """Enhanced expert/gate meta-optimizer with sparsity and entropy-based strategies.
    
    Features:
    - Entropy-based gate LR adjustment
    - Sparsity target enforcement
    - Expert specialization encouragement
    - Dynamic parameter group handling
    """
    
    def __init__(self,
                 update_frequency: int = 1,
                 gate_lr_factor: float = 0.1,
                 expert_lr_factor: float = 1.0,
                 entropy_threshold: float = 0.8,
                 entropy_adjustment_factor: float = 0.5,
                 sparsity_target: float = 0.7,
                 sparsity_tolerance: float = 0.1,
                 specialization_bonus: float = 1.2):
        """
        Args:
            update_frequency: How often to update hyperparameters
            gate_lr_factor: Base LR multiplier for gate parameters
            expert_lr_factor: Base LR multiplier for expert parameters
            entropy_threshold: Gate entropy below which to adjust LR
            entropy_adjustment_factor: Factor for entropy-based LR adjustment
            sparsity_target: Target sparsity level (fraction of experts used)
            sparsity_tolerance: Tolerance around sparsity target
            specialization_bonus: LR bonus for well-specialized experts
        """
        super().__init__(update_frequency)
        self.gate_lr_factor = gate_lr_factor
        self.expert_lr_factor = expert_lr_factor
        self.entropy_threshold = entropy_threshold
        self.entropy_adjustment_factor = entropy_adjustment_factor
        self.sparsity_target = sparsity_target
        self.sparsity_tolerance = sparsity_tolerance
        self.specialization_bonus = specialization_bonus
        
        # State tracking
        self._entropy_history = []
        self._sparsity_history = []
        self._expert_usage_history = []
    
    def update_hparams(self, metrics: Dict[str, float], step: int) -> Dict[str, Any]:
        """Update hyperparameters with sophisticated expert/gate logic."""
        base_lr = metrics.get('lr', 1e-3)
        gate_entropy = metrics.get('gate_entropy', 1.0)
        gate_sparsity = metrics.get('gate_sparsity', 0.5)
        expert_usage = metrics.get('expert_usage', None)  # List of usage rates per expert
        
        # Track metrics
        self._entropy_history.append(gate_entropy)
        self._sparsity_history.append(gate_sparsity)
        if expert_usage is not None:
            self._expert_usage_history.append(expert_usage)
        
        # Limit history
        for history in [self._entropy_history, self._sparsity_history, self._expert_usage_history]:
            while len(history) > 10:
                history.pop(0)
        
        new_hparams = {}
        
        # 1. Gate Learning Rate Adjustment
        gate_lr = base_lr * self.gate_lr_factor
        
        # Entropy-based adjustment
        if gate_entropy < self.entropy_threshold:
            # Low entropy = confident = reduce exploration rate
            gate_lr *= self.entropy_adjustment_factor
        
        # Sparsity-based adjustment
        if len(self._sparsity_history) >= 3:
            recent_sparsity = sum(self._sparsity_history[-3:]) / 3
            sparsity_error = recent_sparsity - self.sparsity_target
            
            if abs(sparsity_error) > self.sparsity_tolerance:
                if sparsity_error > 0:  # Too sparse
                    gate_lr *= 1.3  # Increase exploration
                else:  # Not sparse enough
                    gate_lr *= 0.7  # Reduce exploration, encourage specialization
        
        # 2. Expert Learning Rate Adjustment
        expert_lr = base_lr * self.expert_lr_factor
        
        # Specialization bonus
        if len(self._expert_usage_history) >= 2:
            recent_usage = self._expert_usage_history[-1]
            if recent_usage is not None and len(recent_usage) > 0:
                # Reward experts that are being used consistently
                usage_variance = sum((u - sum(recent_usage)/len(recent_usage))**2 for u in recent_usage) / len(recent_usage)
                if usage_variance > 0.1:  # High variance = good specialization
                    expert_lr *= self.specialization_bonus
        
        # 3. Dynamic Weight Decay for Gates vs Experts
        base_wd = metrics.get('weight_decay', 0.0)
        
        # Gates need more regularization to prevent overfitting to current expert set
        gate_wd = base_wd * 1.5
        expert_wd = base_wd  # Experts can be more flexible
        
        new_hparams.update({
            'gate_lr': gate_lr,
            'expert_lr': expert_lr,
            'gate_weight_decay': gate_wd,
            'expert_weight_decay': expert_wd
        })
        
        return new_hparams
    
    def reset(self):
        """Reset internal state."""
        super().reset()
        self._entropy_history = []
        self._sparsity_history = []
        self._expert_usage_history = []
    
    def state_dict(self):
        """Return state for checkpointing."""
        return {
            'entropy_history': self._entropy_history,
            'sparsity_history': self._sparsity_history,
            'expert_usage_history': self._expert_usage_history,
            'step_count': self._step_count
        }
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self._entropy_history = state_dict.get('entropy_history', [])
        self._sparsity_history = state_dict.get('sparsity_history', [])
        self._expert_usage_history = state_dict.get('expert_usage_history', [])
        self._step_count = state_dict.get('step_count', 0)