"""
Compression module for MoE model optimization.

This module provides various compression techniques:
- Quantization (PTQ and QAT)
- Pruning (magnitude-based and structured)
- Low-rank approximation
"""

from .quantize import (
    ModelQuantizer,
    PTQQuantizer,
    QATQuantizer,
    quantize_model,
    evaluate_quantized_model
)

from .prune import (
    ModelPruner,
    MagnitudePruner,
    StructuredPruner,
    prune_model,
    evaluate_pruned_model
)

from .lowrank import (
    LowRankApproximator,
    SVDApproximator,
    TuckerApproximator,
    compress_with_lowrank,
    evaluate_lowrank_model
)

__all__ = [
    'ModelQuantizer', 'PTQQuantizer', 'QATQuantizer',
    'quantize_model', 'evaluate_quantized_model',
    'ModelPruner', 'MagnitudePruner', 'StructuredPruner', 
    'prune_model', 'evaluate_pruned_model',
    'LowRankApproximator', 'SVDApproximator', 'TuckerApproximator',
    'compress_with_lowrank', 'evaluate_lowrank_model'
]