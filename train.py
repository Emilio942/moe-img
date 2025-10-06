import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.moe import MoEModel
from monitor.probe import SystemProbe
import os
import json
import math
import time
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for CLI
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

# --- 1. Data Loading & Visualization --- 

RUN_TAG = os.environ.get("RUN_TAG", "")
if RUN_TAG:
    RUN_TAG = RUN_TAG.strip()
    RUN_TAG = "_" + "".join(c for c in RUN_TAG if c.isalnum() or c in ("-","_")).strip("_")

def get_cifar10_loaders(batch_size=64):
    """Downloads CIFAR-10 and returns training and test DataLoaders."""
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def save_heatmap(matrix, epoch, title_prefix='Expert Adjacency Matrix'):
    """Saves a heatmap of the given matrix."""
    output_dir = "reports/graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    if _HAS_SNS:
        sns.heatmap(matrix, annot=True, cmap='viridis', fmt='.2f', vmin=-2, vmax=2)
    else:
        im = plt.imshow(matrix, cmap='viridis', vmin=-2, vmax=2)
        plt.colorbar(im)
    plt.title(f'{title_prefix} - Epoch {epoch+1}')
    plt.xlabel('Expert')
    plt.ylabel('Expert')
    epoch_str = f"{epoch+1:02d}" if epoch >= 0 else "initial"
    fname = f'adjacency_matrix{RUN_TAG}_epoch_{epoch_str}.png'
    if 'usage' in title_prefix.lower():
        fname = f'expert_usage{RUN_TAG}_epoch_{epoch_str}.png'
    save_path = os.path.join(output_dir, fname)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved heatmap to {save_path}")

def log_routing(epoch, topk_indices, topk_weights, out_dir="reports/routing"):
    # This can be slow, so we only log the first batch
    os.makedirs(out_dir, exist_ok=True)
    tag = RUN_TAG if RUN_TAG else ""
    path = os.path.join(out_dir, f"routing{tag}_epoch_{epoch+1:02d}.jsonl")
    with open(path, 'w') as f:
        bsz = topk_indices.shape[0]
        k = topk_indices.shape[1]
        w = topk_weights if topk_weights is not None else torch.full((bsz, k), 1.0/k)
        w = w.detach().cpu().tolist()
        idx = topk_indices.detach().cpu().tolist()
        for i in range(bsz):
            rec = {"sample": i, "indices": idx[i], "weights": w[i]}
            f.write(json.dumps(rec) + "\n")

def log_team_stats(epoch, all_gate_indices, num_experts, out_dir="reports/routing"):
    os.makedirs(out_dir, exist_ok=True)
    tag = RUN_TAG if RUN_TAG else ""
    path = os.path.join(out_dir, f"team_stats{tag}_epoch_{epoch+1:02d}.json")

    if not all_gate_indices:
        return

    # Concatenate all indices from all batches
    full_indices = torch.cat(all_gate_indices, dim=0)
    
    # --- Team Size Distribution ---
    # For now, team size is fixed by top_k, but this could change
    team_sizes = full_indices.shape[1]
    team_size_dist = {team_sizes: full_indices.shape[0]}

    # --- Expert Usage Frequency ---
    expert_counts = torch.zeros(num_experts, dtype=torch.long)
    for i in range(num_experts):
        expert_counts[i] = (full_indices == i).sum()
    
    total_selections = expert_counts.sum()
    if total_selections > 0:
        expert_freq = expert_counts / total_selections
    else:
        expert_freq = torch.zeros(num_experts, dtype=torch.float)

    stats = {
        "epoch": epoch + 1,
        "team_size_distribution": team_size_dist,
        "expert_usage_frequency": expert_freq.tolist()
    }

    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved team stats to {path}")

    # Save expert usage as a heatmap
    save_heatmap(expert_freq.reshape(1, -1).numpy(), epoch, title_prefix='Expert Usage Frequency')


def pairwise_cosine_diversity(selected_outputs):
    B, k, F = selected_outputs.shape
    if k < 2:
        return torch.tensor(0.0, device=selected_outputs.device, dtype=selected_outputs.dtype)
    x = nn.functional.normalize(selected_outputs, dim=-1)
    sims = torch.matmul(x, x.transpose(1,2))
    mask = torch.ones((k, k), device=x.device, dtype=x.dtype) - torch.eye(k, device=x.device, dtype=x.dtype)
    sims = sims * mask
    denom = k * (k - 1)
    return sims.sum(dim=(1,2)).mean() / denom

def entropy_loss(weights):
    eps = 1e-9
    w = weights.clamp_min(eps)
    ent = -(w * (w + eps).log()).sum(dim=-1)
    k = w.shape[-1]
    norm = math.log(k + eps)
    return (ent / max(norm, eps)).mean()

def main():
    """Main function to run the joint training with CIFAR-10."""
    # Hyperparameters
    input_dim = 3 * 32 * 32
    output_dim = 10
    num_experts = 8
    top_k = 3
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = int(os.environ.get("NUM_EPOCHS", 10))
    l1_lambda = float(os.environ.get("L1_LAMBDA", 0.1))
    diversity_lambda = float(os.environ.get("DIVERSITY_LAMBDA", 0.05))
    budget_lambda = float(os.environ.get("BUDGET_LAMBDA", 0.01))
    warmup_epochs = int(os.environ.get("WARMUP_EPOCHS", 2))

    print("--- Loading CIFAR-10 Data ---")
    trainloader, testloader = get_cifar10_loaders(batch_size)

    print("--- Initializing Model and Optimizer for Joint Training ---")
    model = MoEModel(input_dim, output_dim, num_experts, top_k)
    gate_mode = os.environ.get("GATE_MODE", "topk")
    if gate_mode == "rl":
        from rl.rl_agent import ReinforceAgent
        agent = ReinforceAgent(input_dim, num_experts, learning_rate)
    else:
        agent = None

    model.gate.mode = gate_mode
    adj_params = [model.expert_graph.adjacency_matrix]
    other_params = [p for n,p in model.named_parameters() if p is not model.expert_graph.adjacency_matrix]
    adj_lr_factor = 0.1
    print(f"Adjacency matrix LR factor: {adj_lr_factor}")
    optimizer = optim.AdamW([
        {"params": other_params, "lr": learning_rate},
        {"params": adj_params, "lr": learning_rate * adj_lr_factor},
    ])
    criterion = nn.CrossEntropyLoss()
    # Monitoring controls
    MONITOR_ENABLED = os.environ.get("MONITOR_ENABLED", "1") != "0"
    probe = SystemProbe() if MONITOR_ENABLED else None

    print("\nInitial Adjacency Matrix:")
    print(model.expert_graph.adjacency_matrix.data)
    save_heatmap(model.expert_graph.adjacency_matrix.data.clone().cpu().numpy(), epoch=-1)

    print("\n--- Starting Training Loop ---")
    print(f"L1: {l1_lambda}, Diversity: {diversity_lambda}, Budget: {budget_lambda}, Warmup: {warmup_epochs} epochs")
    
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        model.use_adjacency = epoch >= warmup_epochs
        model.gate.top_k = 1 if epoch < warmup_epochs else top_k
        
        running_loss = 0.0
        epoch_gate_indices = []
        epoch_measurements = []
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.view(inputs.size(0), -1) # Flatten
            inputs.requires_grad = True

            optimizer.zero_grad()
            tic = time.perf_counter()
            if MONITOR_ENABLED and probe is not None:
                probe.start(model, inputs)

            if agent is not None:
                # RL-based gating
                gate_indices = [agent.select_action(inputs[i]) for i in range(inputs.shape[0])]
                gate_indices = torch.tensor(gate_indices).unsqueeze(1)
                outputs, gate_logits = model(inputs, gate_indices=gate_indices)
            else:
                # Standard gating
                outputs, gate_logits = model(inputs)

            main_loss = criterion(outputs, labels)
            l1_reg = l1_lambda * torch.norm(model.expert_graph.adjacency_matrix, p=1)

            expert_outputs = torch.stack([expert(inputs) for expert in model.experts], dim=1)
            if agent is None:
                gate_indices, _ = model.gate(inputs)
            epoch_gate_indices.append(gate_indices.detach().cpu())
            k = gate_indices.shape[1]
            idx_expanded = gate_indices.unsqueeze(-1).expand(-1, -1, outputs.shape[-1])
            selected_outputs = expert_outputs.gather(1, idx_expanded)
            topk_weights = getattr(model.gate, 'last_topk_weights', None)
            if topk_weights is None:
                topk_weights = torch.full((inputs.shape[0], k), 1.0/k, device=inputs.device, dtype=inputs.dtype)

            div_loss = pairwise_cosine_diversity(selected_outputs)
            bud_loss = entropy_loss(topk_weights)
            total_loss = main_loss + l1_reg + diversity_lambda * div_loss + budget_lambda * bud_loss

            total_loss.backward()
            optimizer.step()

            # Measurements
            if MONITOR_ENABLED and probe is not None:
                measurements = probe.stop()
            else:
                # Lightweight baseline timing without probing memory/FLOPs
                toc = time.perf_counter()
                time_ms = (toc - tic) * 1000.0
                measurements = {
                    'time_ms': time_ms,
                    'mem_mb': 0.0,
                    'flops': 0,
                    'energy_mj': None,
                }
            epoch_measurements.append(measurements)

            if agent is not None:
                # Calculate reward
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                accuracy = 100 * correct / labels.size(0)
                cost = measurements['time_ms'] + measurements['mem_mb'] # Simple cost for now
                reward = accuracy - 0.1 * cost
                agent.rewards.append(reward)
                agent.finish_episode()

            running_loss += total_loss.item()
        
        # --- Evaluation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.view(images.size(0), -1)
                outputs, _ = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        
        # --- Logging ---
        def _avg(key):
            vals = [m.get(key) for m in epoch_measurements if m.get(key) is not None]
            return sum(vals) / len(vals) if vals else None
        avg_time_ms = _avg('time_ms')
        avg_mem_mb = _avg('mem_mb')
        avg_flops = _avg('flops')
        avg_energy_mj = _avg('energy_mj')
        avg_reward = sum(agent.rewards) / len(agent.rewards) if agent is not None and agent.rewards else 0

        def _fmt(val, fmt):
            return fmt.format(val) if val is not None else 'n/a'

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, "
            f"Accuracy: {accuracy:.2f}%, Avg Time: {_fmt(avg_time_ms, '{:.2f}')}ms, "
            f"Avg Mem: {_fmt(avg_mem_mb, '{:.4f}')}MB, Avg FLOPs: {_fmt(avg_flops, '{:.2f}')}, "
            f"Avg Energy: {_fmt(avg_energy_mj, '{:.4f}')}mJ, Avg Reward: {avg_reward:.2f}"
        )

        # Log path can be customized; default differs for ON/OFF to simplify comparisons
        default_log = f"reports/monitor_{'on' if MONITOR_ENABLED else 'off'}{RUN_TAG}.jsonl"
        MONITOR_LOG_PATH = os.environ.get("MONITOR_LOG_PATH", default_log)
        os.makedirs(os.path.dirname(MONITOR_LOG_PATH), exist_ok=True)
        with open(MONITOR_LOG_PATH, "a") as f:
            log_entry = {
                "epoch": epoch + 1,
                "loss": running_loss / len(trainloader),
                "accuracy": accuracy,
                "avg_time_ms": avg_time_ms,
                "avg_mem_mb": avg_mem_mb,
                "avg_flops": avg_flops,
                "avg_energy_mj": avg_energy_mj,
                "avg_reward": avg_reward
            }
            f.write(json.dumps(log_entry) + "\n")

        # Save checkpoint if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/expert_graph_best.ckpt")
            print(f"Saved new best model with accuracy: {accuracy:.2f}%")

        save_heatmap(model.expert_graph.adjacency_matrix.data.clone().cpu().numpy(), epoch)
        log_team_stats(epoch, epoch_gate_indices, num_experts)

        # Only log routing for the first batch of the last epoch to save time
        if epoch == num_epochs - 1:
            inputs, _ = next(iter(testloader))
            inputs = inputs.view(inputs.size(0), -1)
            _, gate_logits = model(inputs)
            gate_indices, topk_weights = model.gate(inputs)
            log_routing(epoch, gate_indices, topk_weights)

    print("\n--- Training Finished ---")
    print("\nFinal Adjacency Matrix:")
    print(model.expert_graph.adjacency_matrix.data)

    # Optional: compare overhead if reference log is provided
    ref_log = os.environ.get("MONITOR_COMPARE_WITH")
    if ref_log and os.path.exists(ref_log):
        try:
            from reports.compare_monitor_overhead import plot_overhead
            on_log = os.environ.get("MONITOR_LOG_PATH", f"reports/monitor_{'on' if MONITOR_ENABLED else 'off'}{RUN_TAG}.jsonl")
            # If current run is OFF, treat ref as ON and swap
            if not MONITOR_ENABLED:
                summary = plot_overhead(on_log_path=ref_log, off_log_path=on_log)
            else:
                summary = plot_overhead(on_log_path=on_log, off_log_path=ref_log)
            print("Monitor overhead summary:", json.dumps(summary))
        except Exception as e:
            print("Overhead comparison failed:", e)

if __name__ == "__main__":
    main()