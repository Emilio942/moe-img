import json
import matplotlib.pyplot as plt
import os

def plot_monitor_logs():
    """Plots the monitor logs."""
    log_file = "reports/monitor_logs.jsonl"
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    epochs = []
    losses = []
    accuracies = []
    avg_times = []
    avg_mems = []
    avg_flops = []
    avg_energies = []

    with open(log_file, "r") as f:
        for line in f:
            log_entry = json.loads(line)
            epochs.append(log_entry.get("epoch"))
            losses.append(log_entry.get("loss"))
            accuracies.append(log_entry.get("accuracy"))
            avg_times.append(log_entry.get("avg_time_ms"))
            avg_mems.append(log_entry.get("avg_mem_mb"))
            avg_flops.append(log_entry.get("avg_flops"))
            avg_energies.append(log_entry.get("avg_energy_mj"))

    output_dir = "reports/graphs"
    os.makedirs(output_dir, exist_ok=True)

    # Plot Time/Batch vs. Epoche
    plt.figure()
    plt.plot(epochs, avg_times)
    plt.xlabel("Epoch")
    plt.ylabel("Avg Time (ms)")
    plt.title("Time/Batch vs. Epoch")
    plt.savefig(os.path.join(output_dir, "time_vs_epoch.png"))
    plt.close()

    # Plot Mem/Batch vs. Epoche
    plt.figure()
    plt.plot(epochs, avg_mems)
    plt.xlabel("Epoch")
    plt.ylabel("Avg Mem (MB)")
    plt.title("Mem/Batch vs. Epoch")
    plt.savefig(os.path.join(output_dir, "mem_vs_epoch.png"))
    plt.close()

    # Plot Loss vs. Epoche
    plt.figure()
    plt.plot(epochs, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.savefig(os.path.join(output_dir, "loss_vs_epoch.png"))
    plt.close()

    # Plot Accuracy vs. Epoche
    plt.figure()
    plt.plot(epochs, accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Epoch")
    plt.savefig(os.path.join(output_dir, "accuracy_vs_epoch.png"))
    plt.close()

    epochs_with_flops = [e for e, f in zip(epochs, avg_flops) if f is not None]
    avg_flops_filtered = [f for f in avg_flops if f is not None]
    if avg_flops_filtered:
        plt.figure()
        plt.plot(epochs_with_flops, avg_flops_filtered)
        plt.xlabel("Epoch")
        plt.ylabel("Avg FLOPs")
        plt.title("FLOPs vs. Epoch")
        plt.savefig(os.path.join(output_dir, "flops_vs_epoch.png"))
        plt.close()

    # Plot Energy vs. Epoche
    epochs_with_energy = [e for e, en in zip(epochs, avg_energies) if en is not None]
    avg_energies_filtered = [en for en in avg_energies if en is not None]
    if avg_energies_filtered:
        plt.figure()
        plt.plot(epochs_with_energy, avg_energies_filtered)
        plt.xlabel("Epoch")
        plt.ylabel("Avg Energy (mJ)")
        plt.title("Energy vs. Epoch")
        plt.savefig(os.path.join(output_dir, "energy_vs_epoch.png"))
        plt.close()

    print(f"Saved plots to {output_dir}")

if __name__ == "__main__":
    plot_monitor_logs()
