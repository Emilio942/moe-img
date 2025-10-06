# Compression Analysis Report
Generated on: 2025-09-26 15:35:00

## Executive Summary

This report analyzes 8 compression configurations, 
evaluating trade-offs between model size, accuracy, and inference speed.

## Key Findings

### Best Size Reduction
- **Method**: Magnitude Pruning 20%
- **Size Reduction**: -25.0%
- **Accuracy Drop**: 0.000
- **Compression Ratio**: 0.80x

### Best Accuracy Preservation
- **Method**: Magnitude Pruning 50%
- **Accuracy Drop**: 0.000
- **Size Reduction**: -25.0%

### Best Speedup
- **Method**: Magnitude Pruning 30%
- **Speedup**: 1.26x
- **Size Reduction**: -25.0%

## Detailed Results

| Method | Size Reduction (%) | Accuracy Drop | Compression Ratio | Speedup | Functional |
|--------|-------------------|---------------|-------------------|---------|------------|
| Original | 0.0% | 0.000 | 1.00x | 1.00x | ✅ |
| Magnitude Pruning 20% | -25.0% | 0.000 | 0.80x | 0.69x | ✅ |
| Magnitude Pruning 30% | -25.0% | 0.000 | 0.80x | 1.26x | ✅ |
| Magnitude Pruning 50% | -25.0% | 0.000 | 0.80x | 0.66x | ✅ |
| SVD 70% Rank | 69.8% | 0.209 | 3.31x | 0.00x | ❌ |
| SVD 50% Rank | 74.8% | 0.209 | 3.97x | 0.00x | ❌ |
| SVD 30% Rank | 80.1% | 0.209 | 5.01x | 0.00x | ❌ |
| Structured Pruning 25% | 25.0% | 0.209 | 1.33x | 0.00x | ❌ |

## Recommendations

Based on this analysis:

1. **For accuracy-critical applications**: Use Magnitude Pruning 50% (only 0.000 accuracy drop)

## Files Generated
- `size_vs_accuracy.png`: Size reduction vs accuracy trade-off plot
- `pareto_front.png`: Pareto front analysis
- `compression_heatmap.png`: Performance heatmap
- `compression_results.json`: Raw results data
