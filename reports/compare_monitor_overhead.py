import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_series(rows: List[Dict], key: str) -> List[Tuple[int, float]]:
    series: List[Tuple[int, float]] = []
    for r in rows:
        epoch = r.get('epoch')
        val = r.get(key)
        if epoch is not None and val is not None:
            series.append((int(epoch), float(val)))
    # sort by epoch
    series.sort(key=lambda x: x[0])
    return series


def compute_overhead(on_log_path: str, off_log_path: str) -> Dict[str, float]:
    """Compute median overhead percentages for time, mem, energy.

    Returns dict with keys: time_overhead_pct, mem_overhead_pct, energy_overhead_pct
    """
    on_rows = _read_jsonl(on_log_path)
    off_rows = _read_jsonl(off_log_path)
    keys = {
        'time': 'avg_time_ms',
        'mem': 'avg_mem_mb',
        'energy': 'avg_energy_mj',
    }
    results: Dict[str, float] = {}
    for name, key in keys.items():
        on_series = dict(_extract_series(on_rows, key))
        off_series = dict(_extract_series(off_rows, key))
        common_epochs = sorted(set(on_series.keys()) & set(off_series.keys()))
        if not common_epochs:
            results[f'{name}_overhead_pct'] = float('nan')
            continue
        ratios = []
        for ep in common_epochs:
            off_val = off_series[ep]
            on_val = on_series[ep]
            if off_val <= 0:
                continue
            ratios.append(on_val / off_val)
        if not ratios:
            results[f'{name}_overhead_pct'] = float('nan')
        else:
            ratios.sort()
            median_ratio = ratios[len(ratios)//2]
            results[f'{name}_overhead_pct'] = (median_ratio - 1.0) * 100.0
    return results


def plot_overhead(on_log_path: str, off_log_path: str, out_dir: str = 'reports/graphs') -> Dict[str, float]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    on_rows = _read_jsonl(on_log_path)
    off_rows = _read_jsonl(off_log_path)

    # Extract time series
    on_time = _extract_series(on_rows, 'avg_time_ms')
    off_time = _extract_series(off_rows, 'avg_time_ms')
    on_mem = _extract_series(on_rows, 'avg_mem_mb')
    off_mem = _extract_series(off_rows, 'avg_mem_mb')
    on_energy = _extract_series(on_rows, 'avg_energy_mj')
    off_energy = _extract_series(off_rows, 'avg_energy_mj')

    # Plot time overhead per epoch (ratio)
    def _plot_ratio(on_series, off_series, ylabel, filename):
        on_dict, off_dict = dict(on_series), dict(off_series)
        epochs = sorted(set(on_dict.keys()) & set(off_dict.keys()))
        if not epochs:
            return
        ratios = []
        for ep in epochs:
            o = off_dict[ep]
            ratios.append((ep, (on_dict[ep] / o) if o > 0 else float('nan')))
        xs = [e for e, _ in ratios]
        ys = [r for _, r in ratios]
        plt.figure()
        plt.plot(xs, ys, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs. Epoch (ON/OFF ratio)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()

    _plot_ratio(on_time, off_time, 'Time ratio', 'monitor_overhead_time_ratio.png')
    _plot_ratio(on_mem, off_mem, 'Memory ratio', 'monitor_overhead_mem_ratio.png')
    _plot_ratio(on_energy, off_energy, 'Energy ratio', 'monitor_overhead_energy_ratio.png')

    # Summary bar with median overhead
    summary = compute_overhead(on_log_path, off_log_path)
    labels = ['Time', 'Memory', 'Energy']
    values = [summary.get('time_overhead_pct', float('nan')),
              summary.get('mem_overhead_pct', float('nan')),
              summary.get('energy_overhead_pct', float('nan'))]
    plt.figure()
    bars = plt.bar(labels, values, color=['#4e79a7', '#59a14f', '#e15759'])
    plt.ylabel('Overhead (%)')
    plt.title('Monitor overhead (ON vs OFF)')
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), f"{val:.1f}%", ha='center', va='bottom')
    plt.ylim(0, max([v for v in values if v == v] + [10]) * 1.2)
    plt.grid(axis='y', alpha=0.2)
    plt.savefig(os.path.join(out_dir, 'monitor_overhead_summary.png'))
    plt.close()
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare monitor ON vs OFF logs and plot overhead.')
    parser.add_argument('--on', required=True, help='Path to ON JSONL log')
    parser.add_argument('--off', required=True, help='Path to OFF JSONL log')
    parser.add_argument('--out', default='reports/graphs', help='Output directory for plots')
    args = parser.parse_args()
    res = plot_overhead(args.on, args.off, args.out)
    print(json.dumps(res, indent=2))
