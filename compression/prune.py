"""
Pruning module for model compression.

Provides magnitude-based and structured pruning techniques to reduce
model parameters while maintaining performance.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, Any, Optional, Tuple, Union, List, Set
import copy
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPruner:
    """Base class for model pruning."""
    
    def __init__(self, target_sparsity: float = 0.3, accuracy_threshold: float = 0.02):
        """
        Initialize pruner.
        
        Args:
            target_sparsity: Target sparsity ratio (e.g., 0.3 = 30% of weights removed)
            accuracy_threshold: Maximum acceptable accuracy drop
        """
        self.target_sparsity = target_sparsity
        self.accuracy_threshold = accuracy_threshold
        
    def get_model_sparsity(self, model: nn.Module) -> float:
        """Calculate current sparsity of the model."""
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
            
        return zero_params / total_params if total_params > 0 else 0.0
    
    def get_parameter_count(self, model: nn.Module) -> Dict[str, int]:
        """Get parameter count statistics."""
        total_params = 0
        trainable_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            zero_params += (param == 0).sum().item()
            
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'zero_params': zero_params,
            'sparsity': zero_params / total_params if total_params > 0 else 0.0
        }
    
    def prune(self, model: nn.Module, **kwargs) -> nn.Module:
        """Abstract method for pruning."""
        raise NotImplementedError
        
    def evaluate_pruning(self, original_model: nn.Module, 
                        pruned_model: nn.Module,
                        eval_fn=None) -> Dict[str, Any]:
        """Evaluate pruning results."""
        original_stats = self.get_parameter_count(original_model)
        pruned_stats = self.get_parameter_count(pruned_model)
        
        # Calculate size reduction (assuming float32)
        original_size_mb = (original_stats['total_params'] * 4) / (1024 * 1024)
        pruned_size_mb = ((pruned_stats['total_params'] - pruned_stats['zero_params']) * 4) / (1024 * 1024)
        
        # Evaluate accuracy if eval function provided
        original_acc = eval_fn(original_model) if eval_fn else None
        pruned_acc = eval_fn(pruned_model) if eval_fn else None
        accuracy_drop = (original_acc - pruned_acc) if (original_acc and pruned_acc) else None
        
        return {
            'original_params': original_stats['total_params'],
            'pruned_params': pruned_stats['total_params'] - pruned_stats['zero_params'],
            'sparsity': pruned_stats['sparsity'],
            'original_size_mb': original_size_mb,
            'pruned_size_mb': pruned_size_mb,
            'compression_ratio': original_size_mb / pruned_size_mb if pruned_size_mb > 0 else float('inf'),
            'size_reduction_percent': (1 - pruned_size_mb/original_size_mb) * 100 if original_size_mb > 0 else 0,
            'original_accuracy': original_acc,
            'pruned_accuracy': pruned_acc,
            'accuracy_drop': accuracy_drop,
            'meets_sparsity_target': pruned_stats['sparsity'] >= self.target_sparsity,
            'meets_accuracy_target': accuracy_drop <= self.accuracy_threshold if accuracy_drop else None
        }


class MagnitudePruner(ModelPruner):
    """Global magnitude-based pruning implementation."""
    
    def __init__(self, target_sparsity: float = 0.3, accuracy_threshold: float = 0.02,
                 iterative_steps: int = 3, exclude_bias: bool = True):
        """
        Initialize magnitude pruner.
        
        Args:
            target_sparsity: Target global sparsity
            accuracy_threshold: Maximum acceptable accuracy drop
            iterative_steps: Number of iterative pruning steps
            exclude_bias: Whether to exclude bias parameters from pruning
        """
        super().__init__(target_sparsity, accuracy_threshold)
        self.iterative_steps = iterative_steps
        self.exclude_bias = exclude_bias
        
    def get_prunable_modules(self, model: nn.Module) -> List[Tuple[nn.Module, str]]:
        """Get list of modules and parameters that can be pruned."""
        prunable = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                # Always include weights
                prunable.append((module, 'weight'))
                
                # Include bias if not excluded and bias exists
                if not self.exclude_bias and hasattr(module, 'bias') and module.bias is not None:
                    prunable.append((module, 'bias'))
                    
        return prunable
    
    def calculate_global_threshold(self, model: nn.Module, sparsity: float) -> float:
        """Calculate global magnitude threshold for given sparsity."""
        all_weights = []
        
        # Collect all prunable parameters
        prunable_modules = self.get_prunable_modules(model)
        
        for module, param_name in prunable_modules:
            param = getattr(module, param_name)
            if param is not None:
                all_weights.extend(param.data.abs().flatten().tolist())
        
        if not all_weights:
            return 0.0
            
        # Calculate threshold for desired sparsity
        all_weights = torch.tensor(all_weights)
        threshold_idx = int(sparsity * len(all_weights))
        threshold = torch.kthvalue(all_weights, threshold_idx + 1)[0].item()
        
        return threshold
    
    def apply_global_magnitude_pruning(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Apply global magnitude pruning to model."""
        # Don't copy if model already has pruning masks
        try:
            model_copy = copy.deepcopy(model)
        except RuntimeError:
            # If deepcopy fails (due to existing pruning), work in-place
            model_copy = model
        
        # Calculate global threshold
        threshold = self.calculate_global_threshold(model_copy, sparsity)
        logger.info(f"Applying global magnitude pruning with threshold: {threshold:.6f}")
        
        # Apply pruning to each module
        prunable_modules = self.get_prunable_modules(model_copy)
        
        for module, param_name in prunable_modules:
            # Create custom pruning method based on global threshold
            class GlobalMagnitudePruning(prune.BasePruningMethod):
                PRUNING_TYPE = 'unstructured'
                
                def __init__(self, threshold):
                    self.threshold = threshold
                
                def compute_mask(self, t, default_mask):
                    return torch.abs(t) >= self.threshold
            
            # Apply pruning
            GlobalMagnitudePruning.apply(module, param_name, threshold=threshold)
        
        return model_copy
    
    def prune(self, model: nn.Module, fine_tune_fn=None, **kwargs) -> nn.Module:
        """
        Perform iterative magnitude-based pruning.
        
        Args:
            model: Original model to prune
            fine_tune_fn: Optional function for fine-tuning between pruning steps
            **kwargs: Additional arguments
            
        Returns:
            Pruned model
        """
        logger.info(f"Starting iterative magnitude pruning to {self.target_sparsity:.1%} sparsity...")
        
        current_model = copy.deepcopy(model)
        
        # Calculate sparsity schedule
        sparsity_schedule = np.linspace(0, self.target_sparsity, self.iterative_steps + 1)[1:]
        
        for step, target_sparsity in enumerate(sparsity_schedule):
            logger.info(f"Pruning step {step + 1}/{self.iterative_steps}, "
                       f"target sparsity: {target_sparsity:.1%}")
            
            # Apply pruning
            current_model = self.apply_global_magnitude_pruning(current_model, target_sparsity)
            
            # Fine-tune if function provided
            if fine_tune_fn is not None:
                logger.info("Fine-tuning after pruning step...")
                current_model = fine_tune_fn(current_model)
                
            # Log current sparsity
            current_sparsity = self.get_model_sparsity(current_model)
            logger.info(f"Achieved sparsity: {current_sparsity:.1%}")
        
        logger.info("Magnitude pruning completed.")
        return current_model


class StructuredPruner(ModelPruner):
    """Structured pruning implementation (removes entire channels/filters)."""
    
    def __init__(self, target_sparsity: float = 0.3, accuracy_threshold: float = 0.02,
                 granularity: str = 'filter', importance_metric: str = 'l1'):
        """
        Initialize structured pruner.
        
        Args:
            target_sparsity: Target sparsity (fraction of filters/channels to remove)
            accuracy_threshold: Maximum acceptable accuracy drop
            granularity: Pruning granularity ('filter' or 'channel')
            importance_metric: Metric for importance ranking ('l1', 'l2', 'geometric_median')
        """
        super().__init__(target_sparsity, accuracy_threshold)
        self.granularity = granularity
        self.importance_metric = importance_metric
        
    def calculate_filter_importance(self, weight: torch.Tensor) -> torch.Tensor:
        """Calculate importance scores for filters/channels."""
        if self.importance_metric == 'l1':
            # L1 norm of each filter
            if len(weight.shape) == 4:  # Conv2d: [out_channels, in_channels, H, W]
                importance = torch.sum(torch.abs(weight), dim=(1, 2, 3))
            elif len(weight.shape) == 2:  # Linear: [out_features, in_features]
                importance = torch.sum(torch.abs(weight), dim=1)
            else:
                importance = torch.sum(torch.abs(weight), dim=tuple(range(1, len(weight.shape))))
        
        elif self.importance_metric == 'l2':
            # L2 norm of each filter
            if len(weight.shape) == 4:
                importance = torch.norm(weight, p=2, dim=(1, 2, 3))
            elif len(weight.shape) == 2:
                importance = torch.norm(weight, p=2, dim=1)
            else:
                importance = torch.norm(weight, p=2, dim=tuple(range(1, len(weight.shape))))
        
        elif self.importance_metric == 'geometric_median':
            # Geometric median-based importance (simplified)
            if len(weight.shape) == 4:
                # Flatten each filter and compute geometric median distance
                flattened = weight.view(weight.size(0), -1)
                median = torch.median(flattened, dim=0)[0]
                importance = torch.norm(flattened - median, p=2, dim=1)
            else:
                # Fallback to L2 for non-conv layers
                importance = torch.norm(weight, p=2, dim=tuple(range(1, len(weight.shape))))
        
        else:
            raise ValueError(f"Unknown importance metric: {self.importance_metric}")
            
        return importance
    
    def prune_conv_layer(self, conv_layer: nn.Conv2d, sparsity: float) -> nn.Conv2d:
        """Prune convolutional layer by removing filters."""
        weight = conv_layer.weight.data
        importance_scores = self.calculate_filter_importance(weight)
        
        # Calculate number of filters to remove
        num_filters = weight.size(0)
        num_to_remove = int(num_filters * sparsity)
        
        if num_to_remove >= num_filters:
            logger.warning(f"Cannot remove {num_to_remove} filters from layer with only {num_filters} filters")
            num_to_remove = max(0, num_filters - 1)  # Keep at least one filter
        
        if num_to_remove == 0:
            return conv_layer
        
        # Get indices of least important filters
        _, sorted_indices = torch.sort(importance_scores)
        filters_to_remove = sorted_indices[:num_to_remove]
        filters_to_keep = sorted_indices[num_to_remove:]
        
        # Create new layer with reduced filters
        new_out_channels = num_filters - num_to_remove
        new_conv = nn.Conv2d(
            in_channels=conv_layer.in_channels,
            out_channels=new_out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=conv_layer.bias is not None,
            padding_mode=conv_layer.padding_mode
        )
        
        # Copy weights for kept filters
        new_conv.weight.data = weight[filters_to_keep]
        if conv_layer.bias is not None:
            new_conv.bias.data = conv_layer.bias.data[filters_to_keep]
            
        return new_conv
    
    def prune_linear_layer(self, linear_layer: nn.Linear, sparsity: float) -> nn.Linear:
        """Prune linear layer by removing output features."""
        weight = linear_layer.weight.data
        importance_scores = self.calculate_filter_importance(weight)
        
        # Calculate number of features to remove
        num_features = weight.size(0)
        num_to_remove = int(num_features * sparsity)
        
        if num_to_remove >= num_features:
            logger.warning(f"Cannot remove {num_to_remove} features from layer with only {num_features} features")
            num_to_remove = max(0, num_features - 1)
        
        if num_to_remove == 0:
            return linear_layer
        
        # Get indices of least important features
        _, sorted_indices = torch.sort(importance_scores)
        features_to_remove = sorted_indices[:num_to_remove]
        features_to_keep = sorted_indices[num_to_remove:]
        
        # Create new layer with reduced features
        new_out_features = num_features - num_to_remove
        new_linear = nn.Linear(
            in_features=linear_layer.in_features,
            out_features=new_out_features,
            bias=linear_layer.bias is not None
        )
        
        # Copy weights for kept features
        new_linear.weight.data = weight[features_to_keep]
        if linear_layer.bias is not None:
            new_linear.bias.data = linear_layer.bias.data[features_to_keep]
            
        return new_linear
    
    def prune(self, model: nn.Module, layer_sparsities: Optional[Dict[str, float]] = None, 
              **kwargs) -> nn.Module:
        """
        Perform structured pruning on the model.
        
        Args:
            model: Original model to prune
            layer_sparsities: Per-layer sparsity ratios (optional)
            **kwargs: Additional arguments
            
        Returns:
            Pruned model with modified architecture
        """
        logger.info(f"Starting structured pruning with {self.granularity} granularity...")
        
        model_copy = copy.deepcopy(model)
        
        # If no per-layer sparsities provided, use global sparsity
        if layer_sparsities is None:
            layer_sparsities = {}
        
        # Recursively prune modules
        for name, module in model_copy.named_modules():
            if isinstance(module, nn.Conv2d):
                sparsity = layer_sparsities.get(name, self.target_sparsity)
                if sparsity > 0:
                    logger.info(f"Pruning Conv2d layer '{name}' with {sparsity:.1%} sparsity")
                    pruned_layer = self.prune_conv_layer(module, sparsity)
                    
                    # Replace the layer in the model
                    parent_name = '.'.join(name.split('.')[:-1])
                    layer_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent_module = dict(model_copy.named_modules())[parent_name]
                    else:
                        parent_module = model_copy
                        
                    setattr(parent_module, layer_name, pruned_layer)
                    
            elif isinstance(module, nn.Linear):
                sparsity = layer_sparsities.get(name, self.target_sparsity)
                if sparsity > 0:
                    logger.info(f"Pruning Linear layer '{name}' with {sparsity:.1%} sparsity")
                    pruned_layer = self.prune_linear_layer(module, sparsity)
                    
                    # Replace the layer in the model
                    parent_name = '.'.join(name.split('.')[:-1])
                    layer_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent_module = dict(model_copy.named_modules())[parent_name]
                    else:
                        parent_module = model_copy
                        
                    setattr(parent_module, layer_name, pruned_layer)
        
        logger.info("Structured pruning completed.")
        return model_copy


# Utility functions
def prune_model(model: nn.Module, method: str = 'magnitude', **kwargs) -> nn.Module:
    """
    Convenience function to prune a model.
    
    Args:
        model: Model to prune
        method: Pruning method ('magnitude' or 'structured')
        **kwargs: Arguments for the pruner
        
    Returns:
        Pruned model
    """
    if method.lower() == 'magnitude':
        pruner = MagnitudePruner(**kwargs)
    elif method.lower() == 'structured':
        pruner = StructuredPruner(**kwargs)
    else:
        raise ValueError(f"Unknown pruning method: {method}. Use 'magnitude' or 'structured'.")
    
    return pruner.prune(model, **kwargs)


def evaluate_pruned_model(original_model: nn.Module, 
                         pruned_model: nn.Module,
                         eval_fn=None,
                         target_sparsity: float = 0.3) -> Dict[str, Any]:
    """
    Evaluate pruning results.
    
    Args:
        original_model: Original unpruned model
        pruned_model: Pruned model
        eval_fn: Function to evaluate model accuracy
        target_sparsity: Target sparsity threshold
        
    Returns:
        Dictionary with evaluation metrics
    """
    pruner = ModelPruner(target_sparsity)
    return pruner.evaluate_pruning(original_model, pruned_model, eval_fn)


def remove_pruning_masks(model: nn.Module) -> nn.Module:
    """
    Permanently remove pruning masks and actually delete pruned weights.
    
    Args:
        model: Model with pruning masks
        
    Returns:
        Model with masks removed
    """
    for module in model.modules():
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')
        if hasattr(module, 'bias_mask'):
            prune.remove(module, 'bias')
    
    return model


def get_layer_wise_sparsity(model: nn.Module) -> Dict[str, float]:
    """
    Calculate sparsity for each layer in the model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary mapping layer names to sparsity ratios
    """
    layer_sparsities = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            if hasattr(module, 'weight') and module.weight is not None:
                total_params = module.weight.numel()
                zero_params = (module.weight == 0).sum().item()
                sparsity = zero_params / total_params if total_params > 0 else 0.0
                layer_sparsities[name] = sparsity
    
    return layer_sparsities