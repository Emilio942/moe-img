import csv, json, time, random
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from rl.rl_env import MoEGatingEnv

class FlatBaselineRunner:
    """Baseline without learning: random expert (or top_k random set) selection each episode.
    Produces CSV compatible subset of RLRunner columns (missing ema fields)."""
    def __init__(self, env: MoEGatingEnv, episodes: int = 50, out_dir: str='reports/rl_runs', run_name: Optional[str]=None, seed: Optional[int]=None):
        self.env=env
        self.episodes=episodes
        if seed is not None:
            random.seed(seed); torch.manual_seed(seed)
        self.out_dir=Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        ts=int(time.time())
        self.run_name = run_name or f'flat_{ts}'
        self.csv_path=self.out_dir / f'{self.run_name}.csv'
        self.jsonl_path=self.out_dir / f'{self.run_name}.jsonl'
        self.meta_path=self.out_dir / f'{self.run_name}_meta.json'
        self.rows: List[Dict[str,Any]]=[]

    def run(self, log_every: int=10):
        state,_ = self.env.reset()
        for ep in range(1,self.episodes+1):
            # random action(s)
            if hasattr(self.env.action_space,'nvec') and len(self.env.action_space.nvec)>1:
                action=[random.randrange(n) for n in self.env.action_space.nvec]
            else:
                action=random.randrange(self.env.action_space.n)
            _, reward, terminated, truncated, info = self.env.step(action)
            row={
                'episode': ep,
                'reward': reward,
                'accuracy': info.get('accuracy'),
                'cost': info.get('cost'),
                'action': info.get('action'),
                'entropy': None
            }
            self.rows.append(row)
            if ep % log_every==0:
                print(f"[FlatBaseline] Ep {ep}/{self.episodes} reward={reward:.4f} acc={row['accuracy']:.4f} cost={row['cost']:.4f}")
            if terminated or truncated:
                state,_=self.env.reset()
        self._write()
        return self.rows

    def _write(self):
        headers=['episode','reward','accuracy','cost','action','entropy']
        with self.csv_path.open('w', newline='') as f:
            w=csv.DictWriter(f, fieldnames=headers); w.writeheader(); [w.writerow(r) for r in self.rows]
        with self.jsonl_path.open('w') as f:
            for r in self.rows: f.write(json.dumps(r)+'\n')
        # meta (reuse RL pareto logic quick inline)
        pts=sorted(self.rows, key=lambda d:(d['cost'],-d['accuracy']))
        pareto=[]; best=-1
        for p in pts:
            if p['accuracy']>best:
                pareto.append(p); best=p['accuracy']
        meta={'episodes': self.episodes,'pareto': pareto,'csv': str(self.csv_path),'jsonl': str(self.jsonl_path)}
        with self.meta_path.open('w') as f: json.dump(meta,f,indent=2)

__all__=['FlatBaselineRunner']
