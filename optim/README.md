# Meta-Optimizer Module

This module provides dynamic hyperparameter adjustment capabilities for PyTorch optimizers during training.

## Components

### MetaOptimizer (Abstract Base Class)
- `update_hparams(metrics, step)` - Core method for hyperparameter updates
- `should_update(step)` - Determines when to apply updates based on frequency
- `reset()` - Reset internal state

### OptimizerWrapper
- Wraps any PyTorch optimizer
- Integrates with MetaOptimizer instances
- Automatically applies hyperparameter updates during training
- Pass-through interface maintains compatibility

### Implementations

#### PlateauMetaOptimizer
- **Strategy**: Heuristic-based approach
- **Features**:
  - Learning rate reduction on loss plateaus
  - Weight decay increase on high gradient norms
  - Configurable patience and reduction factors

#### ExpertGateMetaOptimizer  
- **Strategy**: Component-specific learning rates
- **Features**:
  - Separate learning rates for gate vs expert parameters
  - Gate LR adjustment based on entropy (exploration vs exploitation)
  - Supports MoE architecture training patterns

## Usage Example

```python
import torch.optim as optim
from optim.meta_optimizer import OptimizerWrapper, PlateauMetaOptimizer

# Setup
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
meta_opt = PlateauMetaOptimizer(patience=3, lr_factor=0.5)
wrapped_optimizer = OptimizerWrapper(optimizer, meta_opt)

# Training loop
for epoch in range(num_epochs):
    # ... forward pass, loss calculation ...
    
    wrapped_optimizer.zero_grad()
    loss.backward()
    
    # Calculate metrics
    metrics = {
        'loss': loss.item(),
        'lr': wrapped_optimizer.param_groups[0]['lr'],
        'grad_norm': torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    }
    
    # Update hyperparameters and step
    wrapped_optimizer.update_hyperparameters(metrics)
    wrapped_optimizer.step()
```

## Available Meta-Optimizers

### 1. PlateauMetaOptimizer
Simple heuristic meta-optimizer that reduces learning rate on loss plateaus.

**Features:**
- Learning rate reduction when validation loss plateaus
- Weight decay adjustment based on gradient norms

### 2. ExpertGateMetaOptimizer
Treats expert and gate parameters differently with separate learning rates.

**Features:**
- Separate LRs for gate vs expert parameters
- Entropy-based gate LR adjustment

### 3. AdvancedPlateauMetaOptimizer (NEW)
Enhanced plateau detection with multiple heuristic strategies.

**Features:**
- Multi-criteria plateau detection
- Weight norm monitoring for overfitting detection
- Gradient norm-based weight decay adjustment
- Loss trend analysis
- Warmup and cooldown phases

### 4. SmartExpertGateMetaOptimizer (NEW)
Enhanced expert/gate optimizer with sparsity and entropy-based strategies.

**Features:**
- Entropy-based gate LR adjustment
- Sparsity target enforcement
- Expert specialization encouragement
- Dynamic parameter group handling

### 5. RLMetaOptimizer (NEW)
RL-based meta-optimizer using REINFORCE to learn optimal hyperparameter adjustments.

**Features:**
- Policy network learns from training dynamics
- State representation from loss trends, gradient norms, cost metrics
- Action space for LR multiplication and weight decay adjustment
- REINFORCE training with baseline
- Entropy regularization for exploration

**Example:**
```python
meta_opt = RLMetaOptimizer(
    update_frequency=10,
    lr_actions=[0.5, 1.0, 2.0],
    wd_actions=[-1e-5, 0.0, 1e-5],
    policy_lr=1e-4,
    entropy_weight=0.01
)
```

## Configuration

All meta-optimizers support:
- `update_frequency`: How often to check for updates (default: every step)
- Component-specific parameters for each strategy

**Advanced Configuration Examples:**

```python
# Advanced plateau detection with warmup
meta_opt = AdvancedPlateauMetaOptimizer(
    patience=3,
    lr_factor=0.5,
    grad_norm_threshold=10.0,
    weight_decay_factor=1.5,
    warmup_steps=50,
    trend_window=5
)

# Smart expert/gate handling
meta_opt = SmartExpertGateMetaOptimizer(
    gate_lr_factor=0.1,
    expert_lr_factor=1.0,
    entropy_threshold=0.8,
    sparsity_target=0.7,
    specialization_bonus=1.2
)
```

## Future Extensions

- RL-based meta-optimizer using policy gradients
- Multi-objective optimization (accuracy vs computational cost)
- Adaptive update frequencies based on training phase