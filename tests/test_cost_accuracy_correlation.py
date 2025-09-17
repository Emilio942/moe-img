import csv, os, tempfile
from rl.rl_analysis import load_runs, plot_cost_accuracy_correlation


def test_cost_accuracy_correlation_plot_creation(tmp_path):
    # Create synthetic CSV with simple linear relation accuracy = 1 - 0.1*cost noise-free
    csv_path = tmp_path / 'run.csv'
    fieldnames = ['episode','reward','accuracy','cost','entropy']
    rows = []
    for ep in range(10):
        cost = ep * 0.1 + 0.5  # increasing cost
        acc = 1.0 - 0.05 * ep   # decreasing accuracy => strong negative correlation
        rows.append({'episode': ep, 'reward': acc - 0.1*cost, 'accuracy': acc, 'cost': cost, 'entropy': 0.0})
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    loaded = load_runs(str(csv_path))
    out_dir = tmp_path / 'plots'
    corr = plot_cost_accuracy_correlation(loaded, str(out_dir))
    assert os.path.exists(out_dir / 'cost_accuracy_correlation.png')
    # Expect strong negative correlation near -1
    assert corr < -0.95
