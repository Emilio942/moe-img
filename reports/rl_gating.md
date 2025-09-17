# Analyse des RL-Gatings (Phase 4)

Dieses Dokument analysiert die Ergebnisse des Trainings mit RL-basiertem Gating.

## 1. Reward-Kurven (roh & geglättet)

Plotted files:
* `reward_curve.png` (roher Episoden-Reward)
* `ema_reward_curve.png` (Exponentiell geglättet, α=0.1)
* `moving_avg_reward_curve.png` (gleitendes Fenster, w=5)

Interpretation: EMA zeigt mittelfristigen Trend; Moving Average glättet kurzfristiges Rauschen.

## 2. Policy-Entropie

Plot: `entropy_curve.png`.
Hohe Entropie = Exploration; fallende Entropie kann auf Konvergenz oder Premature Collapse hindeuten.

## 3. Tradeoff-Plot: Accuracy vs. Kosten

Plot: `pareto.png`.
Pareto-Punkte zeigen Episoden-Fronte. Fortschritt = Verschiebung nach oben-links.

## 4. Multi-Run Vergleich

Falls mehrere Runs vorhanden: `reward_compare.png`, `cost_compare.png` zeigen Stabilität & Varianz.
Nächste Schritte: Basis-Top-k Gate Runs hinzufügen für direkte Gegenüberstellung.

## 5. Interpretation (Stub)

Vorläufig: EMA- und Moving-Average-Kurven erleichtern Trendanalyse bei volatilen Rewards. Weitere Runs nötig für statistische Aussagekraft.
