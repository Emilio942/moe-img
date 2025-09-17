from pathlib import Path
from typing import List, Dict, Any, Sequence, Iterable, Tuple
import csv
import json
import math
import numpy as np

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt


def load_runs(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Convert numeric fields
            for k in ['episode','reward','accuracy','cost','entropy']:
                if k in r and r[k] not in (None,''):
                    r[k] = float(r[k]) if k != 'episode' else int(float(r[k]))
            rows.append(r)
    return rows


def plot_curves(rows: Sequence[Dict[str, Any]], out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    eps = [r['episode'] for r in rows]
    # Include extended metrics if present (ema_reward, moving_avg_reward)
    base_metrics = ['reward','accuracy','cost','entropy']
    extended = []
    if any('ema_reward' in r for r in rows):
        extended.append('ema_reward')
    if any('moving_avg_reward' in r for r in rows):
        extended.append('moving_avg_reward')
    metrics = base_metrics + extended
    for m in metrics:
        if any(m not in r or r[m] is None for r in rows):
            continue
        ys = [r[m] for r in rows]
        plt.figure(figsize=(4,3))
        plt.plot(eps, ys, label=m)
        plt.xlabel('Episode')
        plt.ylabel(m.capitalize())
        plt.title(f'{m} vs Episode')
        plt.tight_layout()
        out_path = Path(out_dir) / f'{m}_curve.png'
        plt.savefig(out_path)
        plt.close()


def plot_pareto(rows: Sequence[Dict[str, Any]], out_dir: str):
    # Extract cost vs accuracy points
    pts = [(r['cost'], r['accuracy']) for r in rows if r.get('cost') is not None and r.get('accuracy') is not None]
    if not pts:
        return
    # Pareto front: sort by cost asc, keep points with strictly increasing accuracy
    pts_sorted = sorted(pts, key=lambda x: (x[0], -x[1]))
    pareto = []
    best_acc = -math.inf
    for c, a in pts_sorted:
        if a > best_acc:
            pareto.append((c,a))
            best_acc = a
    costs, accs = zip(*pts)
    p_costs, p_accs = zip(*pareto)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4,3))
    plt.scatter(costs, accs, s=18, alpha=0.6, label='Episodes')
    plt.plot(p_costs, p_accs, color='red', marker='o', label='Pareto front')
    plt.xlabel('Cost (normalized)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy-Cost Pareto')
    plt.legend()
    plt.tight_layout()
    out_path = Path(out_dir) / 'pareto.png'
    plt.savefig(out_path)
    plt.close()


def plot_cost_accuracy_correlation(rows: Sequence[Dict[str, Any]], out_dir: str) -> float:
    """Scatter plot cost vs accuracy with Pearson correlation.

    Returns the Pearson correlation coefficient (float). If insufficient variance
    or <2 points, returns 0.0.
    """
    pts: List[Tuple[float,float]] = [
        (r['cost'], r['accuracy'])
        for r in rows
        if r.get('cost') is not None and r.get('accuracy') is not None
    ]
    if len(pts) < 2:
        return 0.0
    costs = np.array([p[0] for p in pts], dtype=float)
    accs = np.array([p[1] for p in pts], dtype=float)
    if costs.std() == 0 or accs.std() == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(costs, accs)[0,1])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4,3))
    plt.scatter(costs, accs, s=22, alpha=0.7, label='Episodes')
    plt.xlabel('Cost')
    plt.ylabel('Accuracy')
    plt.title(f'Cost-Accuracy (r={corr:.2f})')
    plt.tight_layout()
    out_path = Path(out_dir) / 'cost_accuracy_correlation.png'
    plt.savefig(out_path)
    plt.close()
    return corr


def generate_all(csv_path: str, out_dir: str):
    rows = load_runs(csv_path)
    plot_curves(rows, out_dir)
    plot_pareto(rows, out_dir)
    plot_cost_accuracy_correlation(rows, out_dir)
    return True


# --- Multi-run comparison utilities ---
def load_multi(run_csv_paths: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
    data = {}
    for p in run_csv_paths:
        key = Path(p).stem
        data[key] = load_runs(p)
    return data

def compare_runs(run_csv_paths: Iterable[str], out_dir: str, metrics: Sequence[str] = ('reward','cost')):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runs = load_multi(run_csv_paths)
    for metric in metrics:
        plt.figure(figsize=(5,3))
        plotted = False
        for name, rows in runs.items():
            if any(metric not in r or r[metric] is None for r in rows):
                continue
            eps = [r['episode'] for r in rows]
            ys = [r[metric] for r in rows]
            plt.plot(eps, ys, label=name)
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel('Episode')
        plt.ylabel(metric)
        plt.title(f'{metric} comparison')
        plt.legend(fontsize=8)
        plt.tight_layout()
        out_path = Path(out_dir) / f'{metric}_compare.png'
        plt.savefig(out_path)
        plt.close()

__all__ = [
    'load_runs', 'plot_curves', 'plot_pareto', 'plot_cost_accuracy_correlation',
    'generate_all', 'load_multi', 'compare_runs'
]
