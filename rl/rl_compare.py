import json
from pathlib import Path
from typing import Dict, Any
import csv

def _load_csv(path: str):
    rows=[]
    with open(path,'r') as f:
        r=csv.DictReader(f)
        for row in r:
            for k in ['episode','reward','accuracy','cost']:
                if k in row and row[k] not in (None,''):
                    row[k]=float(row[k]) if k!='episode' else int(float(row[k]))
            rows.append(row)
    return rows

def summarize_run(rows):
    if not rows: return {}
    n=len(rows)
    acc=[r['accuracy'] for r in rows if r.get('accuracy') is not None]
    cost=[r['cost'] for r in rows if r.get('cost') is not None]
    reward=[r['reward'] for r in rows if r.get('reward') is not None]
    def _avg(x): return sum(x)/len(x) if x else None
    return {
        'episodes': n,
        'mean_accuracy': _avg(acc),
        'mean_cost': _avg(cost),
        'mean_reward': _avg(reward),
        'best_accuracy': max(acc) if acc else None,
        'min_cost': min(cost) if cost else None,
    }

def compare(baseline_csv: str, rl_csv: str, out_json: str, out_md: str):
    b_rows=_load_csv(baseline_csv)
    r_rows=_load_csv(rl_csv)
    b_sum=summarize_run(b_rows)
    r_sum=summarize_run(r_rows)
    diff={}
    for k in ['mean_accuracy','mean_cost','mean_reward','best_accuracy','min_cost']:
        if b_sum.get(k) is not None and r_sum.get(k) is not None:
            diff[k]=r_sum[k]-b_sum[k]
    result={'baseline': b_sum,'rl': r_sum,'diff': diff}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json,'w') as f: json.dump(result,f,indent=2)
    # Markdown report
    lines=["# Flat vs RL Comparison","","| Metric | Baseline | RL | RL-Baseline |","|---|---|---|---|"]
    def fmt(v):
        return f"{v:.4f}" if isinstance(v,(int,float)) and v is not None else str(v)
    for k,label in [('mean_accuracy','Mean Accuracy'),('mean_cost','Mean Cost'),('mean_reward','Mean Reward'),('best_accuracy','Best Accuracy'),('min_cost','Min Cost')]:
        lines.append(f"| {label} | {fmt(b_sum.get(k))} | {fmt(r_sum.get(k))} | {fmt(diff.get(k))} |")
    Path(out_md).write_text('\n'.join(lines)+"\n")
    return result

__all__=['compare']