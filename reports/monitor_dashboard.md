# Monitor Dashboard (ON vs OFF Overhead)

This dashboard summarizes the runtime overhead introduced by enabling the training monitor.

How to reproduce:
- Produce two JSONL logs with identical training config:
  - OFF (no monitoring): `reports/monitor_off.jsonl`
  - ON (monitoring enabled): `reports/monitor_on.jsonl`
- Each log line must at least contain: `epoch`, `avg_time_ms`, `avg_mem_mb`, and `avg_energy_mj`.

Generate plots and summary:
- Use the script `reports/compare_monitor_overhead.py`:

  ON vs OFF overhead summary will be saved to `reports/graphs/monitor_overhead_summary.png` and per-epoch ratio plots:
  - `monitor_overhead_time_ratio.png`
  - `monitor_overhead_mem_ratio.png`
  - `monitor_overhead_energy_ratio.png`

Acceptance target:
- Monitor overhead < 5% on median for time.

Notes:
- Ensure same seeds, batch sizes, and eval cadence for fair comparison.
- Discard first N warmup batches when computing averages in your logging pipeline.
