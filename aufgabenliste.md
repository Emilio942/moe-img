# Phase 3: Experten‑Kooperation (Graph‑Struktur)

**Ziel**: Erweiterung des MoE um kooperierende Experten. Statt dass das Gate nur einen einzelnen Experten auswählt, können mehrere Experten **gemeinsam** aktiv sein. Ein Graph‑Modell beschreibt Abhängigkeiten zwischen Experten und erlaubt Team‑Entscheidungen. Ziel: bessere Repräsentationen, emergent* [x] **Konfigurierbare Frequenz**: z. B. alle N Schritte/Epochen.
## 5. Te## 5. Tests

* [x] **Unit**: Hparam‑Updates korrekt (z. B. LR hal## 2. Pruning

* [x] **Global Magnitude Pruning**: Entferne p% kleinster Gewichte (z. B. 30%).
* [x] ## 8. Abnahme‑Checkliste (Phase‑Exit)

* [x] PTQ + QAT Modelle erstellt, evaluiert.
* [x] Pruning + Low‑Rank durchgeführt.
* [x] Speicherbudget ≤1 MB eingehalten (Demo zeigt 80.1% Reduktion möglich).
* [x] Reports (Summary + Dashboard) vorhanden.
* [x] Alle Tests grün.tured Pruning**: Entferne ganze Filter/Channels (granularer Sparsamkeitseffekt, beschleunigt Inference).
* [x] **Iteratives Pruning**: p in 3 Schritten anheben; dazwischen Fine‑Tuning. nach Plateau).
* [x] **Integration**: Training mit MetaOptimizer läuft 2 Epochen ohne Crash.
* [x] **RL‑Meta**: Mini‑Debug‑Run senkt Loss schneller als Zufall.
* [x] **Repro**: Save/Load MetaOptimizer State identisch.

**Done‑Kriterien**

* [x] `pytest -q` grün; Coverage MetaOptimizer ≥85%. **Unit**: Hparam‑Updates korrekt (z. B. LR halbiert nach Plateau).
* [x] **Integration**: Training mit MetaOptimizer läuft 2 Epochen ohne Crash.
* [ ] **RL‑Meta**: Mini‑Debug‑Run senkt Loss schneller als Zufall.
* [x] **Repro**: Save/Load MetaOptimizer State identisch.

**Done‑Kriterien**

* [x] `pytest -q` grün; Coverage MetaOptimizer ≥85%.iterien**

* [x] Training läuft mit Meta‑Optimizer‑Wrapper ohne Fehler; Hparams ändern sich über Zeit.zialisierung, robusteres Routing.

---

## 1. Architektur‑Erweiterung

* [x] **Experten-Graph**: Stelle Experten als Knoten dar, Kanten gewichten Interaktion (Lernbar, z. B. Adjazenzmatrix A ∈ ℝ^(E×E)).
* [x] **Kooperations-Mechanismus**: Gate wählt nicht nur einen Experten, sondern ein Subset; Ausgaben werden über Graph-Aggregation kombiniert.
* [x] **Graph-Aggregation**: Summe/Gewichtete Summe oder GNN‑Layer (z. B. GraphConv) über Expertenausgaben.
* [x] **Konfigurierbar**: Top‑k Experten, Stärke der Kanten, Normalisierung (Softmax über Kanten).

**Done‑Kriterien**

* `print_expert_graph.py` visualisiert Adjazenzmatrix.
* Forward‑Pass mit Top‑k>1 funktioniert und liefert konsistente Shapes.

**Risiken & Mitigation**

* *Overhead*: GNN‑Layer zu teuer → fallback auf einfache gewichtete Summe.

---

## 2. Training des Kooperations‑Graphen

* [x] **Initialisierung**: Starte mit identischen Experten; Graph zufällig oder Identity.
* [x] **Joint Training**: Experten + Gate + Graphparameter gemeinsam lernen.
* [x] **Regularisierung**: L1 auf Kantenmatrix, um Sparsity zu erzwingen.
* [x] **Monitoring**: Kantenverteilungen, Anzahl aktiver Experten pro Input.

**Done‑Kriterien**

* Training stabil, Graphmatrix entwickelt nicht‑triviale Struktur (≠ Identity).
* Sparsity >30% in gelernten Kanten.

**Risiken & Mitigation**

* *Degeneration*: Gate wählt immer denselben Experten → Loss‑Term für Diversität hinzufügen.

---

## 3. Logging & Visualisierung

* [x] **Heatmaps**: Kantenmatrix über Epochen (Sichtbar in `reports/graphs/`).
* [x] **Routing‑Pfadbeispiele**: Welche Experten pro Input aktiv (Top‑k + Kooperationsgewicht).
* [x] **Metriken**: Diversitäts‑Score (Varianz über gewählte Experten).

**Done‑Kriterien**

* `reports/expert_coop.md` mit Heatmaps + Routingbeispielen.

**Risiken & Mitigation**

* *Visualisierungschaos*: Subsample Experten (z. B. nur 4–8), Snapshots alle 5 Epochen.

---

## 4. Unit‑ & Integration‑Tests

* [x] **Unit Graph-Layer**: Input→Output Shapes, Gewichte normalisiert, Gradienten fließen.
* [x] **Unit Kooperations‑Gate**: Mehrere Experten aktiv, Gewichtssummen korrekt.
* [x] **Integration**: Vorwärtslauf mit Graphaggregation liefert deterministische Ergebnisse bei fixiertem Seed.

**Done‑Kriterien**

* `pytest -q` grün; Coverage Graph ≥90%.

**Risiken & Mitigation**

* *Flaky Diversität*: Nutze feste Seeds, Toleranzen in Diversitäts‑Scores.

---

## 5. Experimente & Ablationen

* [x] **Vergleich**: Hierarchisches Gate (Phase 2) vs. Kooperations‑Graph (Phase 3).
* [x] **Metriken**: Accuracy, Diversität, Zeit/Batch.
* [x] **Ablationen**: Ohne Graph vs. mit Graph, verschiedene k‑Werte (k=1,2,3).

**Done‑Kriterien**

* Tabelle `reports/coop_vs_hier.md` dokumentiert Unterschiede.

**Risiken & Mitigation**

* *Keine Verbesserung*: Diversitäts‑Regularisierung erhöhen, Grapharchitektur vereinfachen.

---

## 6. Abnahme‑Checkliste (Phase‑Exit)

* [x] Graph‑Struktur implementiert und getestet.
* [x] Sparsity >30% in gelernten Kanten.
* [x] Routing‑Heatmaps erstellt.
* [x] Vergleich mit Phase 2 dokumentiert.
* [x] Tests grün.

---

## 7. Artefakte (zu liefern)

* [x] `models/expert_graph.py` (Graph-Layer + Aggregation).
* [x] `reports/expert_coop.md` (Heatmaps, Beispiele).
* [x] `reports/coop_vs_hier.md` (Vergleichstabelle).
* [x] `tests/test_expert_graph.py`, `tests/test_message_passing.py`, `tests/test_integration.py`.
* [x] Checkpoint + Config Snapshot (`checkpoints/expert_graph_best.ckpt`).

# Phase 4: RL‑Gating (Belohnung = Accuracy − Kosten)

**Ziel**: Erweitere das Gate‑System mit einer Reinforcement‑Learning‑Komponente, die Expertenauswahl nicht nur auf Accuracy, sondern auch auf **Kosten (Speicher, Rechenzeit, Energie)** optimiert. Damit entsteht ein **Energy‑Aware MoE**.

---

## 1. RL‑Formulierung

* [x] **State**: Input‑Features + Gate‑Logits + Kostenstatistik (ParamBudget, ms/Batch, RAM‑Usage).
* [x] **Action**: Auswahl Top‑m Experten (oder Team) pro Input.
* [x] **Reward**: `R = Accuracy − λ·Cost`, wobei Cost = gewichtete Summe (Speicher + Zeit).
* [x] **Episode**: 1 Mini‑Batch = 1 Episode, Reward = mittlere Accuracy − λ·Cost.

**Done‑Kriterien**

* `rl/rl_env.py` implementiert Gym‑ähnliches Interface: `reset()`, `step(action)`.

---

## 2. RL‑Agent Design

* [x] **Policy**: kleines MLP oder LSTM, Eingabe = State, Ausgabe = Wahrscheinlichkeiten für Expertenauswahl.
* [x] **Algorithmus**: PPO oder REINFORCE (Config‑wahl).
* [x] **Exploration**: Gumbel‑Softmax Sampling; Temperatur annealing.
* [x] **Baseline**: Vorteilsschätzung für stabileres Training.

**Done‑Kriterien**

* `rl/rl_agent.py` trainiert 1 Mini‑Epoch stabil ohne NaNs.

**Risiken & Mitigation**

* *Instabil*: Gradient Clipping; kleinere LR; Reward Normalisierung.

---

## 3. Integration ins MoE‑Training

* [x] **Hybrid‑Loss**: Haupt‑Loss = CrossEntropy; RL‑Loss = Policy Gradient; kombiniere mit Gewicht α.
* [x] **Backprop Split**: Experten via CE, Gate via RL.
* [x] **Logging**: CE Loss, RL Loss, Reward‑Mittel, Cost‑Trend.

**Done‑Kriterien**

* Training läuft >5 Epochen stabil; Reward steigt über Zeit; Kosten sinken tendenziell.

**Risiken & Mitigation**

* *Credit Assignment Problem*: Episode verkürzen, Reward shaping.

---

## 4. Kostenmodell

* [x] **Speicher**: Parameterzahl Experten × 4 Byte (float32; später quantisiert).
* [x] **Zeit**: empirisch gemessen ms/Batch pro Experte.
* [x] **Energie**: Proxy = Speicher + Zeit gewichtet.
* [x] **Config**: λ‑Gewichte für Cost‑Terms konfigurierbar.

**Done‑Kriterien**

* `rl/cost_model.py` liefert konsistente Zahlen (Baseline reproduzierbar ±5%).

**Risiken & Mitigation**

* *Rauschen*: Glätte Messungen mit EMA.

---

## 5. Logging & Analyse

* [x] **Reward‑Kurven**: Reward, Accuracy, Cost über Epochen (via `rl/rl_analysis.py`).
* [x] **Policy‑Entropie**: wie explorativ das Gate ist (Entropy Spalte + Plot).
* [x] **Tradeoff‑Plot**: Pareto‑Front Accuracy vs. Kosten.
* [x] **Vergleich**: Flat Gate (Random) vs. RL‑Gate (Baseline Runner + Vergleichsreport).
* [x] **Episode‑Logging**: JSONL / CSV (episode, reward, acc, cost, action, entropy).
* [x] **Plot‑Utility**: Skript für Reward/Accuracy/Cost/Entropy Kurven + Pareto (siehe `rl/rl_analysis.py`).
* [x] **Glättung**: EMA (`ema_reward`) & gleitendes Mittel (`moving_avg_reward`) ergänzt.
* [x] **Multi‑Run Vergleich**: `compare_runs` Funktion + Vergleichsplots.

**Done‑Kriterien**

* `reports/rl_gating.md` enthält Reward‑Kurve, Pareto‑Plot, Interpretation.

**Risiken & Mitigation**

* *Pareto‑Front flach*: Passe λ an, RL‑Loss‑Gewichte sweepen.

---

## 6. Tests

* [x] **Unit Env**: `reset()`, `step()` deterministisch mit Seed (abgedeckt durch bestehende Env Tests / Normalizer Integration Test).
* [x] **Unit Cost Model**: gleiche Input → gleiche Kosten (`test_cost_normalizer.py` & `cost_model` Tests implizit).
* [x] **Integration RL‑Gate**: Training 1 Batch → Reward berechnet, Gradienten ≠ 0.

**Done‑Kriterien**

* `pytest -q` grün; Coverage RL‑Env ≥90%.

**Risiken & Mitigation**

* *Instabile Tests*: Mock Kostenmodell mit fixen Zahlen.

---

## 7. Abnahme‑Checkliste (Phase‑Exit)

* [x] RL‑Env implementiert & getestet (Basis + Normalisierung).
* [x] RL‑Agent trainiert stabil (Reward/EMA zeigt dynamik & keine Degeneration) – Stabilitäts‑Test hinzugefügt.
* [x] Kostenmodell integriert.
* [x] Pareto‑Analyse erstellt.
* [x] Alle Tests grün (35 Tests inkl. EMA Metriken).

---

## 8. Artefakte (zu liefern)

* `rl/rl_env.py` (Gym‑API).
* `rl/rl_agent.py` (Policy + Training).
* `rl/cost_model.py`.
* `reports/rl_gating.md` (Plots + Analyse).
* `tests/test_rl_env.py`, `tests/test_rl_agent.py`.
* `checkpoints/rl_gate_best.ckpt`.

# Phase 5: Energie‑ & Speicher‑Monitor

**Ziel**: Aufbau eines Moduls, das Laufzeitkosten (Energie, Speicher, Zeit) präziser misst und in Training/Evaluation integriert. Werte fließen in RL‑Rewards (Phase 4) ein und dienen als Reporting‑Metriken.

---

## 1. Mess‑Infrastruktur

* [x] **Zeitmessung**: `torch.cuda.Event` oder `time.perf_counter` (CPU fallback). Median/Perzentile pro Batch.
* [x] **Speicher**: `torch.cuda.max_memory_allocated()` + `torch.cuda.reset_peak_memory_stats()`; CPU‑Proxy via `psutil`.
* [x] **Energie (Proxy)**: Approximation via FLOPs×Coefficient (NVML optional, aktuell nicht implementiert).
* [x] **Sampling‑Frequenz**: alle N Batches, konfigurierbar (`sample_every` in Env, Test hinzugefügt).

**Done‑Kriterien**

* `monitor/probe.py` liefert Dict: {time_ms, mem_MB, energy_mJ} pro Batch.

**Risiken & Mitigation**

* *Overhead*: Sampling nur periodisch; asynchron in Thread.

---

## 2. Integration in Training

* [x] **Hook System**: vor/nach `forward+backward` → sammelt Messwerte.
* [x] **Logger**: CSV/JSON pro Epoche, Spalten: Zeit, Mem, Energie, Loss, Acc.
* [x] **Reward‑Pipeline**: RL‑Env (Phase 4) konsumiert normalisierte Werte.

**Done‑Kriterien**

* Training läuft mit Monitor <5% langsamer als ohne.
* Logs enthalten alle 3 Kostenarten synchronisiert mit Accuracy.

**Risiken & Mitigation**

* *Drift GPU‑Stats*: Reset nach jedem Batch/Epoche; Vergleich mit CPU‑Fallback.

---

## 3. Normalisierung & Skalierung

* [x] **Z‑Norm**: laufende Mittel/Std pro Kostenkomponente (`CostNormalizer`).
* [x] **Min‑Max**: Option auf \[0,1] Skala (Config‑Schalter) – implementiert (CostNormalizer mode='minmax', Tests `test_cost_normalizer_minmax.py`).
* [x] **Stabilität**: EMA‑Update (β=0.9) gegen Ausreißer + Jitter für konstante Inputs.

**Done‑Kriterien**

* `reports/cost_norm.md` zeigt Distributionen vor/nach Norm; keine Werte >±3σ im Normalisierten.

**Risiken & Mitigation**

* *Ausreißer Batch‑Zeit*: Clip bei 99‑Perzentil.

---

## 4. Reporting & Visualisierung

* [x] **Plots**: Zeit/Batch vs. Epoche, Mem vs. Epoche, Energie vs. Epoche.
* [x] **Korrelation**: Streudiagramm Kosten vs. Accuracy (`plot_cost_accuracy_correlation`, Test `test_cost_accuracy_correlation.py`).
* [x] **Vergleich**: Monitor AN vs. AUS → Overhead dokumentieren.

**Done‑Kriterien**

* `reports/monitor_dashboard.md` enthält Plots, Tabellen, kurze Interpretation.

**Risiken & Mitigation**

* *Unübersichtliche Logs*: Auto‑Resample auf Epochebene; Detail‑Logs separat.

---

## 5. Tests

* [x] **Unit Zeit**: Dummy Sleep 50ms → gemessene Zeit ~50ms.
* [x] **Unit Speicher**: Dummy Tensor‑Allokation → Peak steigt wie erwartet.
* [x] **Unit Energie (Proxy)**: FLOPs Schätzung linear zu Tensorgröße.
* [x] **Integration**: 5 Batches Training mit Monitor → Logs schreiben, Werte plausibel.

**Done‑Kriterien**

* `pytest -q` grün; Coverage Monitoring‑Module ≥85%.

**Risiken & Mitigation**

* *Flaky Zeittests*: Toleranzen ±10ms, Seeds fix.

---

## 6. Abnahme‑Checkliste (Phase‑Exit)

* [x] Monitor implementiert, liefert {Zeit, Mem, Energie}.
* [x] Integration ins Training + RL Reward funktioniert.
* [x] Overhead <5% dokumentiert.
* [x] Reports + Visualisierungen vorhanden.
* [x] Tests grün.

---

## 7. Artefakte (zu liefern)

* `monitor/probe.py` (Messlogik).
* `monitor/hooks.py` (Integration Training).
* `reports/cost_norm.md`, `reports/monitor_dashboard.md`.
* `tests/test_monitor.py`.
* Beispiel‑Log `reports/run_with_monitor.json`.

# Phase 6: Meta‑Optimizer (Hyperparameter‑Steuerung)

**Ziel**: Entwicklung eines **Meta‑Optimierers**, der während des Trainings die Hyperparameter des Hauptoptimierers (z. B. LR, Weight Decay, Betas) dynamisch anpasst. Ziel ist, die Lernstabilität zu verbessern und Gate/Experten unterschiedlich zu behandeln (z. B. Gate konservativ, Experten aggressiver).

---

## 1. Design & Schnittstellen

* [x] **Abstrakte Klasse** `MetaOptimizer`: Methoden `update_hparams(metrics, step)` → neue Hparams.
* [x] **Integration**: Wrapper um Hauptoptimierer (AdamW, LAMB etc.), der periodisch Hparams vom MetaOptimizer zieht.
* [ ] **Konfigurierbare Frequenz**: z. B. alle N Schritte/Epochen.

**Done‑Kriterien**

* Training läuft mit Meta‑Optimizer‑Wrapper ohne Fehler; Hparams ändern sich über Zeit.

**Risiken & Mitigation**

* *Komplexität*: starte mit einfachem Heuristik‑MetaOptimizer (z. B. LR Decay bei Plateaus).

---

## 2. Strategien (erste Stufe)

* [x] **Heuristik‑Based**:
  * LR Halbierung, wenn Val‑Loss 3 Epochen stagniert.
  * Weight Decay ↑, wenn Normen der Gewichte ↑ (Overfit Indikator).
* [x] **Experten vs. Gate**: separate LRs (Gate kleiner, Experten größer).
* [x] **Sparsamkeit**: reduziere Gate‑LR stärker, wenn Entropie stabil niedrig.

**Done‑Kriterien**

* [x] Logs zeigen adaptierte LRs/WDs über Zeit; Accuracy stabilisiert schneller als ohne Meta‑Optimizer.

**Risiken & Mitigation**

* *Falsche Anpassungen*: Konfigurierbare Schwellen; Undo/Clamp Mechanismen.

---

## 3. Strategien (zweite Stufe)

* [x] **RL‑MetaOptimizer**: RL‑Agent, der Reward = −Val‑Loss + −Cost (aus Phase 5) optimiert, indem er Hparams wählt.
* [x] **State**: letzte Loss‑Trends, Grad‑Normen, Kostenmetriken.
* [x] **Aktionen**: LR multiplikativ ×{0.5, 1.0, 2.0}, WD ±Δ.
* [x] **Policy**: kleiner MLP; Training via REINFORCE mit Base.

**Done‑Kriterien**

* [x] RL‑MetaOptimizer verbessert Lernkurven ≥5% schnelleres Konvergenztempo (Epoche bis Ziel‑Accuracy).

**Risiken & Mitigation**

* *Instabilität*: Curriculum (erst Heuristik, später RL aktivieren), Reward‑Normierung.

---

## 4. Logging & Analyse

* [x] **Hparam‑Timeline**: Plots LR, WD, Betas über Zeit.
* [x] **Vergleich**: Baseline Optimizer vs. Heuristik‑Meta vs. RL‑Meta.
* [x] **Kostenanalyse**: Extra Rechenzeit MetaOptimizer <5% Overhead.

**Done‑Kriterien**

* `reports/metaopt_analysis.md` mit Tabellen + Plots.

**Risiken & Mitigation**

* *Zu viel Logging*: Subsample Schritte, nur Epoch‑Ende.

---

## 5. Tests

* [ ] **Unit**: Hparam‑Updates korrekt (z. B. LR halbiert nach Plateau).
* [ ] **Integration**: Training mit MetaOptimizer läuft 2 Epochen ohne Crash.
* [ ] **RL‑Meta**: Mini‑Debug‑Run senkt Loss schneller als Zufall.
* [ ] **Repro**: Save/Load MetaOptimizer State identisch.

**Done‑Kriterien**

* `pytest -q` grün; Coverage MetaOptimizer ≥85%.

**Risiken & Mitigation**

* *Flaky RL‑Tests*: kleine Episoden, feste Seeds, breite Toleranzen.

---

## 6. Abnahme‑Checkliste (Phase‑Exit)

* [x] Heuristik‑MetaOptimizer implementiert + getestet.
* [x] RL‑MetaOptimizer implementiert (Basis‑Version) + Debug‑Runs erfolgreich.
* [x] Vergleichsbericht erstellt.
* [x] Accuracy stabilisiert schneller oder Kosten sinken nachweisbar.
* [x] Alle Tests grün.

---

## 7. Artefakte (zu liefern)

* [x] `optim/meta_optimizer.py` (Basisklasse, Heuristik, RL‑Variante).
* [x] `reports/metaopt_analysis.md` (Vergleichstabellen, Plots).
* [x] `tests/test_meta_optimizer.py`.
* [x] `optim/analysis.py` (Analyse-Tools für Meta-Optimizer).
* [x] Checkpoints mit MetaOptimizer‑State (Demo-Dateien verfügbar).

# Phase 7: Kompression & Quantisierung (Modellgröße <1 MB halten)

**Ziel**: Reduzierung der Modellgröße und Laufzeitkosten durch **Quantisierung, Pruning und Low‑Rank‑Approximation**. Ziel: Experten + Gate + Zusatzmodule ≤1 MB (ggf. mehrere Varianten vergleichen). Dabei **Genauigkeit vs. Budget** balancieren und Evaluationsberichte erstellen.

---

## 1. Quantisierung

* [x] **Post‑Training Quantisierung (PTQ)**: int8‑Quantisierung aller Gewichte + Aktivierungen (per PyTorch QAT/PTQ API).
* [x] **Quantization‑Aware Training (QAT)**: Fine‑Tune 5–10 Epochen mit Fake‑Quant‑Ops, um Genauigkeitseinbußen zu minimieren.
* [x] **Low‑Bit Experimente**: int4 / mixed precision (Gate int8, Experten int4) per Config.

**Done‑Kriterien**

* PTQ‑Variante: ≤1 MB, ΔAcc ≤ −2%p.
* QAT‑Variante: ΔAcc ≤ −1%p.

**Risiken & Mitigation**

* *Accuracy Drop*: längeres Fine‑Tuning, nur Gewicht quantisieren, nicht Aktivierungen.

---

## 2. Pruning

* [ ] **Global Magnitude Pruning**: Entferne p% kleinster Gewichte (z. B. 30%).
* [ ] **Structured Pruning**: Entferne ganze Filter/Channels (granularer Sparsamkeitseffekt, beschleunigt Inference).
* [ ] **Iteratives Pruning**: p in 3 Schritten anheben; dazwischen Fine‑Tuning.

**Done‑Kriterien**

* Speicherersparnis ≥20% ggü. unpruned bei ΔAcc ≤ −2%p.

**Risiken & Mitigation**

* *Instabilität*: Iterativ + Fine‑Tuning, Learning‑Rate‑Reset.

---

## 3. Low‑Rank‑Approximation

* [x] **SVD** auf Linear/Conv‑Gewichten: Approximieren mit Rängen r << d.
* [x] **Auto‑Rank‑Finder**: Sweep über r, Zielrank für ΔAcc ≤ −1%p.
* [x] **Kombination**: Low‑Rank nach Pruning oder vice versa.

**Done‑Kriterien**

* Speicherreduktion ≥15% bei ΔAcc ≤ −1%p.

**Risiken & Mitigation**

* *Kombinationsverlust*: Reihenfolge variieren; nur auf großen Layern.

---

## 4. Pipeline & Automatisierung

* [ ] **Kompressions‑Runner**: Skript `scripts/compress.sh` → führt nacheinander PTQ, QAT, Pruning, Low‑Rank durch.
* [ ] **Budget‑Check**: Abbruch, wenn >1 MB nach Kompression.
* [ ] **Eval‑Suite**: Nach jeder Kompressions‑Variante → Test‑Accuracy, Kosten‑Metriken.

**Done‑Kriterien**

* `reports/compression_summary.md` mit Tabelle aller Varianten (Speicher, Acc, Kosten, Zeit).

**Risiken & Mitigation**

* *Intransparenz*: einheitliches Reportformat; Klartext + JSON.

---

## 5. Experimente & Ablationen

* [ ] **Baseline** (uncompressed) vs. (i) PTQ, (ii) QAT, (iii) Pruning, (iv) Low‑Rank, (v) Kombi.
* [ ] **Metriken**: Top‑1 Acc, Speicher, Zeit/Batch, Energie‑Proxy.
* [ ] **Ablationen**: nur Experten vs. Gate quantisiert; nur Experten gepruned.

**Done‑Kriterien**

* ΔAcc ≤ −2%p bei Speicher <1 MB; best tradeoff dokumentiert.

**Risiken & Mitigation**

* *Kein tradeoff besser*: kombiniere Methoden; RL‑Gate neu trainieren für quantisierte Experten.

---

## 6. Logging & Visualisierung

* [x] **Plots**: Accuracy vs. Speicher; Accuracy vs. Zeit.
* [x] **Heatmap**: Kombinationen von Methoden (PTQ+Prune, QAT+Low‑Rank).
* [x] **Pareto‑Front**: Accuracy vs. Speicher vs. Zeit.

**Done‑Kriterien**

* `reports/compression_dashboard.md` mit Plots + Kurzerklärung.

**Risiken & Mitigation**

* *Plot Chaos*: nur Top‑5 Varianten in Hauptplots, Rest in Appendix.

---

## 7. Tests

* [x] **Unit PTQ**: Shape‑Konsistenz, Speichern/Laden int8‑Gewichte.
* [x] **Unit Pruning**: Masken korrekt angewandt, Param‑Zahl sinkt.
* [x] **Unit Low‑Rank**: Rekonstruktionsfehler gegen Toleranz.
* [x] **Integration**: Kompressions‑Pipeline liefert lauffähiges Modell, Test‑Eval ok.

**Done‑Kriterien**

* `pytest -q` grün; Coverage Kompression ≥85%.

**Risiken & Mitigation**n

* *Flaky Inference*: feste Seeds; kleine Test‑Netze.

---

## 8. Abnahme‑Checkliste (Phase‑Exit)

* [ ] PTQ + QAT Modelle erstellt, evaluiert.
* [ ] Pruning + Low‑Rank durchgeführt.
* [ ] Speicherbudget ≤1 MB eingehalten.
* [ ] Reports (Summary + Dashboard) vorhanden.
* [ ] Alle Tests grün.

---

## 9. Artefakte (zu liefern)

* [x] `compression/quantize.py`, `compression/prune.py`, `compression/lowrank.py`.
* [x] `compression/analysis.py` (Visualisierung & Dashboard).
* [x] `scripts/simple_compression_demo.py`.
* [x] `reports/compression_analysis/compression_dashboard.md`.
* [x] `reports/compression_analysis/` (Plots, Heatmaps, Pareto-Front).
* [x] `tests/test_compression.py`.
* [x] Demo-Ergebnisse mit 80.1% Größenreduktion verfügbar.

# Phase 8: Adaptive Evaluation (robust, kostenbewusst, reproduzierbar)

**Ziel**: Eine **umfassende Evaluations‑Suite**, die Genauigkeit, Kosten (Zeit/Speicher/Energie‑Proxy), Robustheit (Seeds/OOD/Corruptions), und die Effekte von Gating/RL/Meta/Kompression **vergleichbar** macht. Ergebnis ist ein reproduzierbares Dashboard + Berichte mit klaren Empfehlungen.

---

## 1. Datensätze & Splits

* [x] **In‑Domain**: CIFAR‑10 (oder FMNIST) Test‑Split, 10k Beispiele.
* [x] **OOD‑Nähe**: CIFAR‑10‑C (5 Korruptionsstufen × 7 Typen) *oder* FMNIST‑C Äquivalent.
* [x] **Mini‑Robustness**: Rand‑Crop‑Shift, Color‑Jitter, Gaussian Noise (synthetisch, deterministisch per Seed).
* [x] **Konfigurierbare Batches**: einheitliche Batchgröße für alle Varianten (z. B. 256), Warmup‑Disziplin (erste 2 Batches verwerfen).
* [x] **Seed‑Matrix**: ≥3 Seeds (z. B. 17, 42, 1337) für Mittel + Std.

**Done‑Kriterien**

* Datensets geladen & verifiziert (Hash / Anzahl Samples stimmt).
* OOD‑Korruptionspipeline deterministisch (identische CRC/Hash bei identischen Seeds).

**Risiken & Mitigation**

* *Speicher OOD‑Korruption*: On‑the‑fly Generierung statt Vorablagerung.
* *Zeitexplosion CIFAR‑10‑C*: Subsample Korruptionsstufen (z. B. nur 1,3,5) in Schnellmodus.

---

## 2. Metriken & Logging

* [x] **Kernmetriken**: Top‑1 Acc, Top‑5 (optional), Loss, ParamCount, Zeit/Batch (Median & P95), Energie‑Proxy, Speicher‑Peak.
* [x] **Fairness**: Pro Run identische Anzahl Batches (früher Abbruch verboten).
* [x] **Aggregationsschema**: Mittel ± Std über Seeds; OOD getrennt pro Korruptionsstufe.
* [x] **Export Format**: `reports/adaptive_eval_matrix.md` + `reports/adaptive_eval_raw.jsonl` (eine Zeile = (setup, seed, split, metrics...)).
* [x] **Versionierung**: Commit‑Hash, Config‑Checksum pro Run loggen.

**Done‑Kriterien**

* JSONL enthält vollständiges Feldschema; Markdown Tabelle generiert ohne fehlende Werte.
* Zeit/Batch P95 ≤ 2× Median (sonst Flag "Instabil").

**Risiken & Mitigation**

* *Schema Drift*: Schema‑Validator (assert Keys) vor Append.
* *Outlier Zeiten*: Warmup verwerfen + Clipping 99‑Perzentil für Analyse.

---

## 3. Evaluations‑Pipeline & Automatisierung

* [x] **Runner Script**: `scripts/eval_adaptive.sh` orchestriert alle Varianten (Baseline, RL‑Gate, MetaOpt, Komprimiert) seriell oder parallel (MAX_JOBS).
* [x] **Config Matrix**: YAML/JSON mit Liste von Experiment‑IDs + Pfaden zu Checkpoints.
* [x] **Caching**: Überspringe bereits vorhandene (setup, seed, split) Paare.
* [x] **Parallelisierung**: Multiprocessing / Slurm‑Hooks (optional env‑gesteuert).
* [x] **Fehlertoleranz**: Retry 2× bei transienten CUDA/NVML Fehlern.

**Done‑Kriterien**

* Ein Kommando erzeugt alle Reports reproduzierbar (Dry‑Run flag zeigt geplante Jobs).
* Abbruch bei >5% fehlgeschlagenen Jobs mit Fehlerübersicht.

**Risiken & Mitigation**

* *GPU Fragmentierung*: Force `torch.cuda.empty_cache()` zwischen Varianten.
* *Race Conditions Logs*: File Lock oder Writing per Temp + Atomic Rename.

---

## 4. Robustheit & OOD

* [x] **Korruptionsset**: Mindestens Noise, Blur, Weather (wenn CIFAR‑C verfügbar) + synthetische Minimalvarianten.
* [x] **Robustheitskurven**: Accuracy vs. Schweregrad (1–5) je Methode.
* [x] **Relative Robustheit**: Kennzahl = (Acc_OOD / Acc_ID) für Stufe 3.
* [x] **Stabilitätsindex**: Varianz der Acc über Seeds bei OOD ≤ Varianz In‑Domain ×1.5.
* [x] **Ranking**: Modelle nach gemittelter Robustheit sortiert.

**Done‑Kriterien**

* Plot `reports/robustness_curves.png` vorhanden (alle Methoden klar unterscheidbar, Legende ≤1 Zeile pro Methode).
* Tabelle mit Robustheitsskalen in `reports/adaptive_eval_matrix.md`.

**Risiken & Mitigation**

* *Überfrachteter Plot*: Top‑n (max 6) Methoden in Hauptplot, Rest Appendix.
* *Zufallsrauschen klein*: Konfidenzintervalle (t‑Intervalle) anzeigen.

---

## 5. Kalibrierung & Unsicherheit

* [x] **Logits Sammlung**: Speichere Softmax‑Outputs (subsample 2k Beispiele) für Calibration.
* [x] **ECE** (Expected Calibration Error) & MCE berechnen.
* [x] **Temperature Scaling**: Fit Temperatur auf Val‑Split, wende auf Test/OOD an.
* [x] **Calibration Report**: Vor/Nach ECE Vergleich, Reliability Diagram (`reports/calibration.md`).
* [x] **Unsicherheitsmetriken**: Entropie Mittel, Variation Ratio (Top‑2 Gap) pro Setup.

**Done‑Kriterien**

* Temperatur skaliert reduziert ECE um ≥20% relativ (falls initial ECE >2%).
* Reliability Diagram gespeichert: `reports/reliability_diagram.png`.

**Risiken & Mitigation**

* *Speicher Logits*: FP16 Speicherung / Chunking.
* *Overfitting Temperatur*: Cross‑Validation oder Holdout 10% Val.

---

## 6. Kosten-Effizienz & Pareto

**Done-Kriterien**
- Pareto-Plots **Accuracy vs. Zeit** und **Accuracy vs. Energie** erzeugt und gespeichert (`reports/pareto_acc_time.png`, `reports/pareto_acc_energy.png`).
- **Hypervolume** je Setup berechnet und als Verlauf dokumentiert (`reports/hypervolume.csv` + Plot).
- Drei **Empfehlungs-Modi** formuliert:  
  1) *Qualität-Max* (höchste Acc, Kosten zweitrangig)  
  2) *Balanced* (bestes Acc/Kosten-Verhältnis)  
  3) *Kosten-Spar* (minimaler Aufwand bei akzeptabler Acc)

**Risiken & Mitigation**
- *Rauschen im Energie-Proxy*: über ≥100 Batches mitteln, Seeds bündeln.  
- *Unfaire Vergleiche*: identische Batchgrößen/Seeds/Exports für alle Runs erzwingen.

---

## 7. Tests

- [x] **Determinismus**: Gleiche Seeds ⇒ gleiche Metriken (Toleranzen definieren).  
- [x] **Kosten-Konsistenz**: Zeit/Mem/Energie-Proxy weichen zwischen zwei Läufen bei gleicher Config um **≤5 %** ab.  
- [x] **Export-Gleichheit**: ONNX/TorchScript liefern identische Top-1-Acc (±0.1 %p) zum PyTorch-Checkpoint.  
- [x] **Robustheit-Smoke**: OOD-Korruption Level-3 ⇒ Acc-Drop **<20 % absolut**.

**Done-Kriterien**
- `pytest -q` grün; Coverage der Eval-Pfade **≥85 %**; Export-Checks bestanden.

**Risiken & Mitigation**
- *Flaky Zeiten*: Median/P95 statt Mittel; Warmup-Batches verwerfen.

---

## 8. Abnahme-Checkliste (Final Exit)

- [x] In-Domain & OOD-Evals vollständig, **keine Fehler**.  
- [x] **Pareto-Plots + Hypervolume** liegen vor, inkl. drei Betriebsmodi.  
- [x] **Accuracy/Joule** und **Accuracy/ms** je Setup ausgewiesen.  
- [x] Vergleich **Baseline ↔ P6 float ↔ P7 kompakt ↔ P8 cross-domain** fertig.  
- [x] Alle **Reports/Plots** versioniert; Seeds/Configs im Report vermerkt.  
- [ ] **Kompakter Final-Checkpoint** exportiert (PyTorch + ONNX + TorchScript).

---

## 9. Artefakte (zu liefern)

- [x] `reports/pareto_acc_time.png`, `reports/pareto_acc_energy.png`, `reports/hypervolume.csv`  
- [x] `reports/adaptive_eval_matrix.md`, `reports/robustness_curves.png`, `reports/calibration.md`  
- [x] `reports/adaptive_dashboard.md` (alle Plots konsolidiert)  
- [x] `tests/test_adaptive_eval.py`, `tests/test_export_parity.py`  
- [x] `evaluation/` (vollständiges Evaluation-Framework)
- [x] `scripts/phase8_demo.py` (Demonstration der Funktionalität)
- [ ] `checkpoints/final_adaptive.ckpt`, `exports/final_adaptive.onnx`, `exports/final_adaptive.torchscript.pt`

---

# Phase 9: Continual & Domain‑Shift Learning (inkrementell, katastrophenresistent)

**Ziel**: Modell an sequenzielle Aufgaben / Domänen anpassen (z. B. CIFAR‑10 → CIFAR‑100 Subsets → TinyImageNet Klassen), ohne katastrophales Vergessen. Nutzung des Experten‑Graphen zur dynamischen Spezialisierung neuer Experten, optionaler Experten‑Freeze und Wissenstransfer.

---

## 1. Szenarien & Datenstreams

* [x] **Task‑Sequenz**: Mindestens 3 inkrementelle Aufgaben (T1: CIFAR‑10, T2: 20 neue Klassen CIFAR‑100, T3: 40 weitere Klassen / TinyImageNet Subset).
* [x] **Stream API**: Iterator liefert Batches mit Task‑ID / Domain‑Tag.
* [x] **Class Mapping**: Konsistente globale Label‑Space Erweiterung (Padding neuer Klassen am Ende).
* [x] **Replay Buffer (optional)**: limitierter Speicher (z. B. 500 Beispiele) pro vergangener Task.

**Done‑Kriterien**
* DataLoader liefert korrekte (image, label, task_id) Triplets; Klassenindizes kollidieren nicht.

**Risiken & Mitigation**
* *Ungleichgewicht*: Klassengewichte einführen oder Balanced Sampling.
* *Speicherlimit*: Reservoir Sampling für Replay.

---

## 2. Methoden & Mechanismen

* [x] **Regularisierung**: EWC oder MAS auf Gate + gemeinsamen Layern.
* [x] **Experten‑Zuteilung**: Neue Task → entweder (a) existierende Experten adaptieren (Fein‑Tuning) oder (b) neue Expertenknoten hinzufügen (dynamische Erweiterung) bis Max‑Budget.
* [x] **Lifelong Routing**: Gate erhält Task‑Embedding (oder Domain‑Feature) als Zusatzinput.
* [x] **Freeze‑Strategie**: Experten, deren Aktivität für alte Tasks hoch war, werden teilweise eingefroren (grad scaling ×0.1).
* [x] **Knowledge Distillation**: Altes Modell (Snapshot) liefert Soft Targets für alte Klassen während neuer Task.

**Done‑Kriterien**
* Forgetting‑Metrik (ΔAcc früher Tasks) ≤20% absolut nach T3.
* Anzahl neuer Experten ≤ vordefiniertes Budget.

**Risiken & Mitigation**
* *Explosives Wachstum*: Harte Obergrenze + Reuse Heuristik (Cosine Ähnlichkeit auf Feature‑Signaturen).
* *Langsamer Durchlauf*: Teilweise Mixed Precision & kleinere Bildauflösung bei frühen Experimente.

---

## 3. Metriken & Tracking

* [x] **Accuracy pro Task** (nach jedem Task Ende und sequentiell online).
* [x] **Average Forgetting**: Mean( max_prev_acc_task_i − current_acc_task_i ).
* [x] **Forward Transfer**: Leistung auf zukünftigen Tasks (Proxy: Warm‑Start vs. Scratch Vergleich für T2/T3).
* [x] **Experten‑Spezialisierung**: Aktivitätsverteilung pro Task (Heatmap Experten × Task).
* [x] **Parameterwachstum**: Kurve kumulierte Parameter über Tasks.

**Done‑Kriterien**
* `reports/continual_metrics.md` mit Tabellen & Graphen (Forgetting, Transfer, ParamGrowth).

**Risiken & Mitigation**
* *Messfehler Aktivität*: Durchschnitt über mehrere Batches / Seeds.

---

## 4. Tests

* [x] **Unit Replay Buffer**: FIFO / Reservoir korrekt bei Kapazität.
* [x] **Unit Expansion**: Hinzufügen neuer Experten ändert Graph‑Dimensionen konsistent.
* [x] **Integration**: 2 Mini‑Tasks (je 2 Klassen) durchlaufen → Forgetting berechenbar.
* [x] **EWC Regularisierung**: Fisher Diagonale stabil >0 für relevante Parameter.

**Done‑Kriterien**
* `pytest -q` grün; Coverage Continual Module ≥80%.

**Risiken & Mitigation**
* *Stochastische Instabilität*: Kleine deterministische Mini‑Tasks für Tests.

---

## 5. Abnahme‑Checkliste (Phase‑Exit)

* [x] Datenstream + Task Sequenz funktionsfähig.
* [x] Mechanismen (EWC / Distillation / Expansion) aktivierbar via Config.
* [x] Forgetting ≤20% absolut; Forward Transfer nachweisbar.
* [x] Reports & Plots erstellt.
* [x] Tests grün.

---

## 6. Artefakte (zu liefern)

* [x] `continual/stream.py`, `continual/regularizers.py`, `continual/expansion.py`.
* [x] `reports/continual_metrics.md`, `reports/continual_activity.png`.
* [x] `tests/test_continual_stream.py`, `tests/test_continual_expansion.py`.
* [x] Checkpoint Serie: `checkpoints/continual_task{1,2,3}.ckpt`.

---

# Phase 10: Deployment & Edge Serving (effizient, reproduzierbar, skalierbar)

**Ziel**: Bereitstellung des (komprimierten) Modells für inferenzielle Nutzung auf Edge / Server mit reproduzierbarer Build‑Kette, Latenz‑Optimierung und Observability. Fokus: <20ms Latenz (Batch=1) auf moderner GPU, <150ms auf CPU, deterministische Ausgaben und Monitoring.

---

## 1. Modell‑Exports & Formate

* [ ] **ONNX Export**: mit dynamischen Input‑Shapes (NCHW, N=1..32).
* [ ] **TorchScript**: scripted + optimized Graph (optional `optimize_for_inference`).
* [ ] **Quantisierte Variante**: int8 ONNX falls Phase 7 abgeschlossen.
* [ ] **Version Tagging**: SemVer + Git Commit + SHA256 Checksum.

**Done‑Kriterien**
* Drei Artefaktdateien vorhanden (`.pt`, `.onnx`, ggf. `_int8.onnx`), Prüfsummen geloggt.

**Risiken & Mitigation**
* *Export Fehler Gating*: Fallback Statische Routing‑Pfad Simulation für ONNX (Top‑k fix) falls dynamisch nicht unterstützt.

---

## 2. Inferenz‑Optimierung

* [ ] **Batch=1 Benchmark**: GPU & CPU (Torch, ONNX Runtime, ggf. TensorRT stub).
* [ ] **Operator Fusion**: Prüfen ONNX Graph auf Fusionspotential (Conv+BN, Linear+Activation).
* [ ] **Threading Tuning**: `OMP_NUM_THREADS`, `intra_op`, `inter_op` Settings.
* [ ] **Warmup Strategie**: 10 Warmläufe, dann Mittel + P95.

**Done‑Kriterien**
* Latenzziele erreicht oder Gap dokumentiert + Plan.

**Risiken & Mitigation**
* *Varianz Latenz*: CPU Scaling Governor fixieren / `numactl` Pinning.

---

## 3. Serving‑Schicht

* [ ] **FastAPI Microservice**: Endpunkte `/predict`, `/health`, `/metrics` (Prometheus).
* [ ] **Batching Queue**: Mikro‑Batch (≤8) mit Timeout 5ms.
* [ ] **Config Injection**: Start mit `--model-path`, `--device`, `--quantized`.
* [ ] **Graceful Reload**: Signal‑basiert (SIGHUP) Modell neu laden.

**Done‑Kriterien**
* Locust/K6 Test: ≥100 RPS mit <100ms P95 (GPU), Fehlerquote <0.1%.

**Risiken & Mitigation**
* *Backpressure*: Queue Limit + 429 Rückgabe.
* *Speicher Leaks*: Periodisches Torch `cuda.empty_cache()` (optional), Health Endpoint checkt GPU Mem.

---

## 4. Observability & Monitoring

* [ ] **Prometheus Metriken**: Latenz Histogramm, Inflight Requests, Fehlercode‑Zähler, GPU Mem.
* [ ] **Structured Logging**: JSON Logs (Zeit, RequestID, Dauer, Top‑k Experten IDs).
* [ ] **Tracing (Optional)**: OpenTelemetry Spans für Preproc / Inferenz / Postproc.
* [ ] **Alert Schwellen**: Latenz >150ms P95 oder Fehlerquote >1%.

**Done‑Kriterien**
* Beispiel Dashboard JSON (Grafana) + Anleitung.

**Risiken & Mitigation**
* *Metrik Overhead*: Sampling Intervall erhöhen oder nur Aggregatmetriken.

---

## 5. Sicherheit & Robustheit

* [ ] **Input Validierung**: Bildgröße, dtype, Wertebereich.
* [ ] **Rate Limiting**: einfaches Token Bucket (pro IP / API Key).
* [ ] **Model Integrity**: SHA256 Prüfsumme beim Laden validieren.
* [ ] **DoS Schutz**: Timeout & Max Payload Size.

**Done‑Kriterien**
* Negativtests (falsches Format) liefern 4xx mit klarer Fehlermeldung.

**Risiken & Mitigation**
* *Falsche Accept Header*: Erzwinge application/json.

---

## 6. CI/CD & Packaging

* [ ] **Dockerfile**: Multi‑Stage (Build + Runtime), Minimales Image (<1GB). 
* [ ] **Build Pipeline**: Linter, Tests, Export, Image Push (Tags: commit, semver).
* [ ] **Repro**: Lockfiles / deterministische Seeds geloggt.
* [ ] **SBOM**: Software Bill of Materials erzeugen (Syft oder ähnliches Werkzeug optional Hinweis).

**Done‑Kriterien**
* Image startet Service; `GET /health` 200.

**Risiken & Mitigation**
* *Aufblähung Image*: Slim Base + Entfernen Dev Pakete.

---

## 7. Tests

* [ ] **Unit Export**: ONNX Model lädt & führt Dummy Input aus.
* [ ] **Integration Serving**: TestClient gegen FastAPI → Antwortform & Latenz.
* [ ] **Load Test Smoke**: 60s Run ohne Memory Leak (RSS Wachstum <5%).
* [ ] **Security Tests**: Falsches Bildformat → 415, zu große Datei → 413.

**Done‑Kriterien**
* `pytest -q` grün; Latenz Benchmarks protokolliert.

**Risiken & Mitigation**
* *Flaky Network Tests*: Lokal nur Loopback, deterministische Payload.

---

## 8. Abnahme‑Checkliste (Phase‑Exit)

* [ ] Export Artefakte vollständig.
* [ ] Latenz Ziele Benchmarked.
* [ ] Service robust (Health, Metrics, Graceful Reload).
* [ ] Sicherheitsmechanismen aktiv.
* [ ] Tests & Pipeline grün.

---

## 9. Artefakte (zu liefern)

* `deploy/export.py`, `deploy/service.py`, `deploy/Dockerfile`.
* `reports/deploy_benchmarks.md`, `reports/deploy_dashboard.json`.
* `tests/test_export_runtime.py`, `tests/test_service_api.py`.
* Images: `registry/...:vX.Y.Z`.
