"""
Continual & Domain-Shift Learning Module

This module implements incremental learning capabilities with catastrophic forgetting prevention
for the Mixture of Experts architecture. It provides:

1. Task sequencing and data streaming
2. Regularization methods (EWC, MAS)
3. Expert expansion and specialization
4. Knowledge distillation
5. Continual learning metrics and tracking

Components:
- stream: Data streaming and task management
- regularizers: EWC/MAS regularization methods
- expansion: Dynamic expert graph expansion
- distillation: Knowledge distillation for old tasks
- metrics: Continual learning evaluation metrics
- trainer: Integrated continual learning trainer
"""

from .stream import (
    ContinualDataStream,
    TaskSequence,
    ReplayBuffer,
    TaskDataLoader,
)

from .regularizers import (
    EWCRegularizer,
    MASRegularizer,
    RegularizerConfig,
)

from .expansion import (
    ExpertExpander,
    ExpansionStrategy,
    ExpansionConfig,
    ExpertReuseMechanism,
)

from .distillation import (
    KnowledgeDistiller,
    DistillationConfig,
    SoftTargetGenerator,
)

from .metrics import (
    ContinualMetrics,
    ForgettingTracker,
    TransferAnalyzer,
    SpecializationTracker,
)

from .trainer import (
    ContinualTrainer,
    ContinualConfig,
    TaskCallback,
)

__all__ = [
    # Data streaming
    'ContinualDataStream',
    'TaskSequence', 
    'ReplayBuffer',
    'TaskDataLoader',
    
    # Regularization
    'EWCRegularizer',
    'MASRegularizer',
    'RegularizerConfig',
    
    # Expert expansion
    'ExpertExpander',
    'ExpansionStrategy',
    'ExpansionConfig',
    'ExpertReuseMechanism',
    
    # Knowledge distillation
    'KnowledgeDistiller',
    'DistillationConfig',
    'SoftTargetGenerator',
    
    # Metrics and tracking
    'ContinualMetrics',
    'ForgettingTracker',
    'TransferAnalyzer',
    'SpecializationTracker',
    
    # Training
    'ContinualTrainer',
    'ContinualConfig',
    'TaskCallback',
]

__version__ = "0.1.0"