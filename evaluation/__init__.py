"""
Adaptive Evaluation Suite - Phase 8
=====================================

Comprehensive evaluation framework for robust, cost-aware, and reproducible
model evaluation across multiple domains, corruption levels, and seeds.

Components:
-----------
- datasets.py: In-domain, OOD, corruption datasets
- metrics.py: Core metrics (accuracy, time, memory, energy proxy)
- pipeline.py: Evaluation orchestration and automation
- robustness.py: OOD evaluation and corruption handling
- calibration.py: Model calibration and uncertainty quantification
- cost_analysis.py: Pareto-optimal analysis and cost-efficiency
- adaptive_runner.py: Main evaluation runner with parallelization
- export_utils.py: Model export (ONNX, TorchScript) and validation

Key Features:
-------------
✅ Multi-seed reproducibility with deterministic evaluation
✅ OOD corruption datasets (CIFAR-10-C style)
✅ Cost-aware Pareto analysis (Accuracy vs Time/Energy)
✅ Calibration metrics (ECE, MCE) with temperature scaling
✅ Hypervolume computation for multi-objective optimization
✅ Export parity validation (PyTorch ↔ ONNX ↔ TorchScript)
✅ Comprehensive logging and dashboard generation
"""

__version__ = "1.0.0"

# Main evaluation components
from .datasets import (
    EvaluationDatasets,
    CorruptionTransforms,
    create_ood_splits
)

from .metrics import (
    CoreMetrics,
    CostMetrics,
    RobustnessMetrics,
    CalibrationMetrics
)

from .pipeline import (
    EvaluationPipeline,
    ExperimentConfig,
    EvaluationRunner
)

from .adaptive_runner import (
    AdaptiveEvaluationRunner,
    run_comprehensive_evaluation
)

__all__ = [
    # Core components
    'EvaluationDatasets',
    'CorruptionTransforms',
    'create_ood_splits',
    'CoreMetrics',
    'CostMetrics', 
    'RobustnessMetrics',
    'CalibrationMetrics',
    'EvaluationPipeline',
    'ExperimentConfig',
    'EvaluationRunner',
    'AdaptiveEvaluationRunner',
    'run_comprehensive_evaluation',
]