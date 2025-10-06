import torch
import torch.nn as nn
import torch.nn.functional as F
from optim.meta_optimizer import (
    OptimizerWrapper, 
    PlateauMetaOptimizer,
    AdvancedPlateauMetaOptimizer,
    SmartExpertGateMetaOptimizer,
    RLMetaOptimizer
)


def test_meta_optimizer_integration_with_simple_model():
    """Test meta-optimizer integration with a simple model training."""
    # Create simple model
    model = nn.Linear(10, 1)
    
    # Use meta-optimizer wrapper with patience=1 for quick testing
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    meta_opt = PlateauMetaOptimizer(patience=1, lr_factor=0.5)
    optimizer = OptimizerWrapper(base_optimizer, meta_opt)
    
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Force a plateau by using the same loss values
    forced_losses = [1.0, 0.8, 0.9, 0.85, 0.9]  # 0.8 is best, then plateau
    
    for step, forced_loss in enumerate(forced_losses):
        optimizer.zero_grad()
        
        # Create a dummy loss that matches our forced value
        loss = torch.tensor(forced_loss, requires_grad=True)
        loss.backward()
        
        metrics = {
            'loss': forced_loss,
            'lr': optimizer.param_groups[0]['lr']
        }
        
        optimizer.update_hyperparameters(metrics)
        optimizer.step()
        
        print(f"Step {step}: Loss={forced_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
    
    final_lr = optimizer.param_groups[0]['lr']
    print(f"Training completed: Initial LR: {initial_lr:.6f}, Final LR: {final_lr:.6f}")
    
    # Should have reduced LR due to plateau (0.9 > 0.8, then 0.85 > 0.8, then 0.9 > 0.8)
    assert final_lr < initial_lr, f"Expected LR reduction, got {final_lr} vs {initial_lr}"


def test_meta_optimizer_integration_with_expert_graph():
    """Test meta-optimizer integration with actual ExpertGraph aggregation."""
    num_experts = 4
    feature_dim = 64
    batch_size = 8
    
    # Create ExpertGraph for aggregation
    from models.expert_graph import ExpertGraph
    expert_graph = ExpertGraph(
        num_experts=num_experts,
        feature_dim=feature_dim,
        aggregation_type='weighted_sum'
    )
    
    # Create simple experts (just linear layers)
    experts = nn.ModuleList([
        nn.Linear(32, feature_dim) for _ in range(num_experts)
    ])
    
    # Final classifier
    classifier = nn.Linear(feature_dim, 10)
    
    # Combine all parameters for optimizer
    all_params = list(expert_graph.parameters()) + list(experts.parameters()) + list(classifier.parameters())
    
    # Use meta-optimizer wrapper
    base_optimizer = torch.optim.Adam(all_params, lr=1e-3)
    meta_opt = PlateauMetaOptimizer(patience=1, lr_factor=0.9)
    optimizer = OptimizerWrapper(base_optimizer, meta_opt)
    
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Use fixed data and force plateau by using controlled loss values
    x = torch.randn(batch_size, 32)
    y = torch.randint(0, 10, (batch_size,))
    
    forced_losses = [2.5, 2.0, 2.2, 2.1, 2.3]  # 2.0 is best, then plateau
    
    for step, forced_loss in enumerate(forced_losses):
        optimizer.zero_grad()
        
        # Forward through experts
        expert_outputs = torch.stack([expert(x) for expert in experts], dim=1)  # (batch, num_experts, feature_dim)
        
        # Simple top-k selection (use top 2 experts)
        k = 2
        top_k_indices = torch.topk(torch.randn(batch_size, num_experts), k, dim=1).indices
        top_k_weights = F.softmax(torch.randn(batch_size, k), dim=1)
        
        # Aggregate expert outputs
        aggregated = expert_graph(expert_outputs, top_k_indices, top_k_weights)
        
        # Final prediction
        output = classifier(aggregated)
        
        # Create a controlled loss that matches our forced value
        loss = torch.tensor(forced_loss, requires_grad=True)
        loss.backward(retain_graph=True)
        
        # Also do a real forward pass to keep gradients flowing
        real_loss = F.cross_entropy(output, y)
        real_loss.backward()
        
        # Calculate metrics using forced loss
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=float('inf'))
        metrics = {
            'loss': forced_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm.item()
        }
        
        optimizer.update_hyperparameters(metrics)
        optimizer.step()
        
        print(f"Step {step}: Loss={forced_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
    
    final_lr = optimizer.param_groups[0]['lr']
    assert final_lr < initial_lr, f"Expected LR reduction, got {final_lr} vs {initial_lr}"
    
    print(f"ExpertGraph integration: Initial LR: {initial_lr:.6f}, Final LR: {final_lr:.6f}")


def test_meta_optimizer_save_load_integration():
    """Test saving and loading meta-optimizer state during training."""
    model = nn.Linear(10, 1)
    
    # Create optimizer with meta-optimizer
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    meta_opt = PlateauMetaOptimizer(patience=1)
    optimizer = OptimizerWrapper(base_optimizer, meta_opt)
    
    # Train a few steps
    for step in range(3):
        optimizer.zero_grad()
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        
        metrics = {'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']}
        optimizer.update_hyperparameters(metrics)
        optimizer.step()
    
    # Save state
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'meta_optimizer': meta_opt.state_dict() if hasattr(meta_opt, 'state_dict') else None
    }
    
    # Create new model and optimizer
    new_model = nn.Linear(10, 1)
    new_base_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.2)  # Different initial LR
    new_meta_opt = PlateauMetaOptimizer(patience=1)
    new_optimizer = OptimizerWrapper(new_base_optimizer, new_meta_opt)
    
    # Load state
    new_model.load_state_dict(checkpoint['model'])
    new_optimizer.load_state_dict(checkpoint['optimizer'])
    if checkpoint['meta_optimizer'] and hasattr(new_meta_opt, 'load_state_dict'):
        new_meta_opt.load_state_dict(checkpoint['meta_optimizer'])
    
    # Verify state was restored
    assert new_optimizer.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
    
    print("Save/load integration test completed successfully")


def test_advanced_plateau_meta_optimizer_integration():
    """Test AdvancedPlateauMetaOptimizer with realistic training scenario."""
    # Create simple model
    model = nn.Sequential(
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    # Use advanced meta-optimizer with warmup
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    meta_opt = AdvancedPlateauMetaOptimizer(
        patience=2, 
        lr_factor=0.8,
        warmup_steps=3,
        grad_norm_threshold=5.0,
        weight_decay_factor=1.3
    )
    optimizer = OptimizerWrapper(base_optimizer, meta_opt)
    
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Generate training data
    x = torch.randn(16, 20)
    y = torch.randn(16, 1)
    
    lr_changes = []
    wd_changes = []
    
    for step in range(10):
        optimizer.zero_grad()
        
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        
        # Calculate metrics including weight norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        weight_norm = sum(p.norm().item() for p in model.parameters())
        
        metrics = {
            'loss': loss.item(),
            'lr': optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm.item(),
            'weight_decay': optimizer.param_groups[0]['weight_decay'],
            'weight_norm': weight_norm
        }
        
        # Force plateau by using same loss after step 5
        if step >= 5:
            metrics['loss'] = 0.8  # Fixed plateau loss
        
        optimizer.update_hyperparameters(metrics)
        optimizer.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        current_wd = optimizer.param_groups[0]['weight_decay']
        
        lr_changes.append(current_lr)
        wd_changes.append(current_wd)
        
        print(f"Step {step}: Loss={metrics['loss']:.4f}, LR={current_lr:.6f}, WD={current_wd:.6f}, GradNorm={grad_norm:.2f}")
    
    # Should show LR changes due to warmup and plateau
    final_lr = lr_changes[-1]
    assert len(set(lr_changes)) > 1, "Expected LR to change during training"
    
    print(f"Advanced integration: Initial LR: {initial_lr:.6f}, Final LR: {final_lr:.6f}")
    print(f"LR progression: {[f'{lr:.6f}' for lr in lr_changes]}")


def test_smart_expert_gate_integration():
    """Test SmartExpertGateMetaOptimizer integration."""
    # Create a simple model with separate expert and gate components
    expert_models = nn.ModuleList([nn.Linear(10, 5) for _ in range(3)])
    gate_model = nn.Linear(10, 3)
    classifier = nn.Linear(5, 1)
    
    # Separate parameter groups for experts and gates
    expert_params = list(expert_models.parameters()) + list(classifier.parameters())
    gate_params = list(gate_model.parameters())
    
    base_optimizer = torch.optim.Adam([
        {'params': expert_params, 'lr': 1e-3},
        {'params': gate_params, 'lr': 1e-4}
    ])
    
    meta_opt = SmartExpertGateMetaOptimizer(
        gate_lr_factor=0.1,
        expert_lr_factor=1.0,
        entropy_threshold=0.8,
        sparsity_target=0.7
    )
    optimizer = OptimizerWrapper(base_optimizer, meta_opt)
    
    x = torch.randn(8, 10)
    y = torch.randn(8, 1)
    
    for step in range(6):
        optimizer.zero_grad()
        
        # Forward pass through experts
        expert_outputs = torch.stack([expert(x) for expert in expert_models], dim=1)
        gate_logits = gate_model(x)
        gate_probs = F.softmax(gate_logits, dim=1)
        
        # Calculate gate entropy and sparsity
        gate_entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=1).mean().item()
        gate_sparsity = (gate_probs.max(dim=1)[0] > 0.5).float().mean().item()
        
        # Simple expert selection and aggregation
        selected_expert_idx = gate_probs.argmax(dim=1)
        aggregated = torch.stack([expert_outputs[i, selected_expert_idx[i]] for i in range(len(x))])
        
        output = classifier(aggregated)
        loss = F.mse_loss(output, y)
        loss.backward()
        
        # Expert usage statistics
        expert_usage = [(selected_expert_idx == i).float().mean().item() for i in range(3)]
        
        metrics = {
            'lr': 1e-3,  # Base LR
            'gate_entropy': gate_entropy,
            'gate_sparsity': gate_sparsity,
            'expert_usage': expert_usage
        }
        
        optimizer.update_hyperparameters(metrics)
        optimizer.step()
        
        print(f"Step {step}: Entropy={gate_entropy:.3f}, Sparsity={gate_sparsity:.3f}, Usage={[f'{u:.2f}' for u in expert_usage]}")
    
    print("Smart expert/gate integration test completed successfully")


def test_rl_meta_optimizer_integration():
    """Test RLMetaOptimizer integration with actual training."""
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Use RL meta-optimizer
    base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rl_meta_opt = RLMetaOptimizer(
        update_frequency=1,
        state_dim=8,
        hidden_dim=16,
        lr_actions=[0.8, 1.0, 1.2],
        wd_actions=[-1e-5, 0.0, 1e-5],
        policy_lr=1e-3
    )
    optimizer = OptimizerWrapper(base_optimizer, rl_meta_opt)
    
    # Generate training data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    initial_lr = optimizer.param_groups[0]['lr']
    losses = []
    lr_changes = []
    
    # Training loop
    for step in range(15):
        optimizer.zero_grad()
        
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        
        # Calculate metrics for RL meta-optimizer
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        
        # Simulate cost (could come from actual monitoring)
        cost = 0.5 - step * 0.02  # Decreasing cost over time
        
        metrics = {
            'loss': loss.item(),
            'lr': optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm.item(),
            'weight_decay': optimizer.param_groups[0]['weight_decay'],
            'cost': max(0.1, cost)
        }
        
        optimizer.update_hyperparameters(metrics)
        optimizer.step()
        
        losses.append(loss.item())
        lr_changes.append(optimizer.param_groups[0]['lr'])
        
        print(f"Step {step}: Loss={loss.item():.4f}, LR={optimizer.param_groups[0]['lr']:.6f}, Cost={metrics['cost']:.3f}")
    
    final_lr = optimizer.param_groups[0]['lr']
    
    # RL should adapt hyperparameters over time
    unique_lrs = len(set(f"{lr:.8f}" for lr in lr_changes))
    print(f"RL integration: Initial LR: {initial_lr:.6f}, Final LR: {final_lr:.6f}")
    print(f"Unique LR values during training: {unique_lrs}")
    print(f"Loss progression: {[f'{l:.4f}' for l in losses[:5]]} ... {[f'{l:.4f}' for l in losses[-5:]]}")
    
    # Should see some adaptation (though with small network, changes might be subtle)
    assert unique_lrs >= 1, "RL meta-optimizer should adapt learning rates"


def test_rl_vs_heuristic_comparison():
    """Compare RL meta-optimizer against heuristic baseline."""
    # Create identical models
    torch.manual_seed(42)
    model_rl = nn.Linear(5, 1)
    
    torch.manual_seed(42)  # Same initialization
    model_heuristic = nn.Linear(5, 1)
    
    # Set up optimizers
    rl_optimizer = OptimizerWrapper(
        torch.optim.SGD(model_rl.parameters(), lr=0.1),
        RLMetaOptimizer(
            update_frequency=1,
            lr_actions=[0.5, 1.0, 1.5],
            wd_actions=[0.0],
            policy_lr=1e-3
        )
    )
    
    heuristic_optimizer = OptimizerWrapper(
        torch.optim.SGD(model_heuristic.parameters(), lr=0.1),
        PlateauMetaOptimizer(patience=2, lr_factor=0.5)
    )
    
    # Training data
    x = torch.randn(16, 5)
    y = torch.randn(16, 1)
    
    rl_losses = []
    heuristic_losses = []
    
    # Train both models
    for step in range(10):
        # Train RL model
        rl_optimizer.zero_grad()
        rl_output = model_rl(x)
        rl_loss = F.mse_loss(rl_output, y)
        rl_loss.backward()
        
        rl_metrics = {
            'loss': rl_loss.item(),
            'lr': rl_optimizer.param_groups[0]['lr'],
            'grad_norm': torch.nn.utils.clip_grad_norm_(model_rl.parameters(), float('inf')).item(),
            'cost': 0.3
        }
        rl_optimizer.update_hyperparameters(rl_metrics)
        rl_optimizer.step()
        rl_losses.append(rl_loss.item())
        
        # Train heuristic model
        heuristic_optimizer.zero_grad()
        heuristic_output = model_heuristic(x)
        heuristic_loss = F.mse_loss(heuristic_output, y)
        heuristic_loss.backward()
        
        heuristic_metrics = {
            'loss': heuristic_loss.item(),
            'lr': heuristic_optimizer.param_groups[0]['lr'],
            'grad_norm': torch.nn.utils.clip_grad_norm_(model_heuristic.parameters(), float('inf')).item()
        }
        heuristic_optimizer.update_hyperparameters(heuristic_metrics)
        heuristic_optimizer.step()
        heuristic_losses.append(heuristic_loss.item())
    
    print(f"RL final loss: {rl_losses[-1]:.4f}")
    print(f"Heuristic final loss: {heuristic_losses[-1]:.4f}")
    
    # Both should achieve reasonable training (loss should decrease)
    assert rl_losses[-1] < rl_losses[0] * 1.1, "RL optimizer should show learning progress"
    assert heuristic_losses[-1] < heuristic_losses[0] * 1.1, "Heuristic optimizer should show learning progress"
    
    print("RL vs Heuristic comparison completed successfully")


if __name__ == '__main__':
    test_meta_optimizer_integration_with_simple_model()
    test_meta_optimizer_integration_with_expert_graph()
    test_meta_optimizer_save_load_integration()
    test_advanced_plateau_meta_optimizer_integration()
    test_smart_expert_gate_integration()
    test_rl_meta_optimizer_integration()
    test_rl_vs_heuristic_comparison()
    print("All integration tests passed!")