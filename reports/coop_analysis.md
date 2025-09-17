# Analyse der Experten-Kooperation (Phase 3)

Dieses Dokument analysiert die Ergebnisse des Trainings mit Experten-Kooperation über einen Graphen.

## 1. Kooperations-Heatmap

Die Kooperationsstärke zwischen den Experten wird durch eine Adjazenzmatrix visualisiert. Die folgenden Heatmaps zeigen die Matrix am Anfang und am Ende des Trainings für einen `k=3` Lauf.

**Initial (Epoche 0):**
![Initial Adjacency Matrix](./graphs/adjacency_matrix_k3_epoch_initial.png)

**Final (Epoche 10):**
![Final Adjacency Matrix](./graphs/adjacency_matrix_k3_epoch_10.png)

Die Matrix wurde am Ende jeder Epoche gespeichert. Man erkennt, wie das Modell lernt, bestimmte Experten-Paare zu bevorzugen.

## 2. Team-Statistik

Die `train.py` Skript erzeugt `team_stats_*.jsonl` Dateien, die die Verteilung der Teamgrößen und die Nutzungshäufigkeit der einzelnen Experten pro Epoche enthalten.

Die folgende Heatmap zeigt die Nutzungshäufigkeit der Experten am Ende eines Laufs ohne Diversitäts-Loss. Man sieht, dass einige Experten deutlich häufiger als andere ausgewählt werden.

![Expert Usage](./graphs/expert_usage_no_diversity_epoch_10.png)

## 3. Ablation Studies

Basierend auf den generierten Graphen wurden verschiedene Ablationsstudien durchgeführt:

*   **`k1` vs. `k3`**: Vergleich der Teamgröße (1 vs. 3).
*   **`no_diversity`**: Training ohne den Diversitäts-Loss.
*   **`no_graph`**: Training ohne den Graphen (Standard MoE).
*   **`no_sparsity`**: Training ohne L1-Regularisierung auf der Adjazenzmatrix.

Ein visueller Vergleich der Heatmaps zeigt, wie diese Bedingungen die gelernte Kooperationsstruktur beeinflussen. Zum Beispiel führt das Entfernen des Sparsity-Loss (`no_sparsity`) zu einer dichteren Adjazenzmatrix.

## 4. Interpretation

Die Visualisierungen zeigen, dass das Modell in der Lage ist, eine nicht-triviale Kooperationsstruktur zwischen den Experten zu lernen. Die Ablationsstudien deuten darauf hin, dass die verschiedenen Loss-Terme (Diversität, Sparsity) und Hyperparameter (k) einen signifikanten Einfluss auf das Endergebnis haben.

Weitere Analysen könnten die Korrelation zwischen der gelernten Graphstruktur und der finalen Modellgenauigkeit untersuchen.
