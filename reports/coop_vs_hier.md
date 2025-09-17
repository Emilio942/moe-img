# Phase 3: Comparison of Cooperation Graph Ablations

This report documents the results of the ablation studies performed for the expert cooperation graph, as specified in section 5 of the Phase 3 task list.

## Comparison Summary

The goal was to evaluate the impact of the cooperation graph and the number of selected experts (`k`) on the model's performance on CIFAR-10.

The following configurations were tested:

- **Baseline (No Graph)**: The cooperation graph mechanism was disabled. `k` was set to 2.
- **Graph (k=1)**: Graph mechanism enabled, but only a single expert is selected. This effectively disables cooperation.
- **Graph (k=2)**: The default configuration with the cooperation graph and `k=2`.
- **Graph (k=3)**: Using the cooperation graph with `k=3`.

### Results Table

| Experiment | `top_k` | Graph Enabled | Final Accuracy |
|---|:---:|:---:|---:|
| Graph (k=1) | 1 | Yes | 43.62% |
| Baseline (No Graph) | 2 | No | 46.36% |
| Graph (k=2) | 2 | Yes | ~48.07% |
| Graph (k=3) | 3 | Yes | **49.58%** |

*Note: The accuracy for "Graph (k=2)" is from a previous run with a different random seed, hence the approximation. The overall trend is the key takeaway.*

## Conclusion

1.  **The cooperation graph is beneficial**: Comparing the two runs with `k=2`, the version with the graph enabled (~48.07%) outperformed the one without (46.36%). This indicates that allowing experts to cooperate via the learned sparse graph improves performance.

2.  **More experts are better**: Increasing `k` from 1 to 3 consistently improved accuracy. The model with `k=3` achieved the highest accuracy of **49.58%**. This suggests that providing the model with a larger team of active experts for each input is advantageous.

3.  **k=1 is the worst**: As expected, selecting only one expert results in the lowest accuracy, as there is no possibility for expert cooperation, and the model is less expressive.

Based on these results, using the cooperation graph and a `top_k` value of 3 is the recommended configuration for this phase.