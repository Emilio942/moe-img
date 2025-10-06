"""
Low-rank approximation module for model compression.

Provides SVD and Tucker decomposition methods to reduce parameter count
while maintaining model performance.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union, List
import copy
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LowRankApproximator:
    """Base class for low-rank approximation methods."""
    
    def __init__(self, rank_ratio: float = 0.5, accuracy_threshold: float = 0.02):
        """
        Initialize low-rank approximator.
        
        Args:
            rank_ratio: Ratio of original rank to keep (e.g., 0.5 = 50% rank reduction)
            accuracy_threshold: Maximum acceptable accuracy drop
        """
        self.rank_ratio = rank_ratio
        self.accuracy_threshold = accuracy_threshold
        
    def get_model_size(self, model: nn.Module) -> Dict[str, int]:
        """Calculate model parameter statistics."""
        total_params = 0
        trainable_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_mb': (total_params * 4) / (1024 * 1024)  # Assuming float32
        }
    
    def decompose(self, model: nn.Module, **kwargs) -> nn.Module:
        """Abstract method for low-rank decomposition."""
        raise NotImplementedError
        
    def evaluate_decomposition(self, original_model: nn.Module, 
                             decomposed_model: nn.Module,
                             eval_fn=None) -> Dict[str, Any]:
        """Evaluate decomposition results."""
        original_stats = self.get_model_size(original_model)
        decomposed_stats = self.get_model_size(decomposed_model)
        
        compression_ratio = original_stats['total_params'] / decomposed_stats['total_params']
        size_reduction = (1 - decomposed_stats['total_params'] / original_stats['total_params']) * 100
        
        # Evaluate accuracy if eval function provided
        original_acc = eval_fn(original_model) if eval_fn else None
        decomposed_acc = eval_fn(decomposed_model) if eval_fn else None
        accuracy_drop = (original_acc - decomposed_acc) if (original_acc and decomposed_acc) else None
        
        return {
            'original_params': original_stats['total_params'],
            'decomposed_params': decomposed_stats['total_params'],
            'original_size_mb': original_stats['size_mb'],
            'decomposed_size_mb': decomposed_stats['size_mb'],
            'compression_ratio': compression_ratio,
            'size_reduction_percent': size_reduction,
            'original_accuracy': original_acc,
            'decomposed_accuracy': decomposed_acc,
            'accuracy_drop': accuracy_drop,
            'meets_accuracy_target': accuracy_drop <= self.accuracy_threshold if accuracy_drop else None
        }


class SVDApproximator(LowRankApproximator):
    """SVD-based low-rank approximation for Linear and Conv2D layers."""
    
    def __init__(self, rank_ratio: float = 0.5, accuracy_threshold: float = 0.02,
                 layer_types: List[str] = None):
        """
        Initialize SVD approximator.
        
        Args:
            rank_ratio: Ratio of original rank to keep
            accuracy_threshold: Maximum acceptable accuracy drop
            layer_types: Types of layers to decompose ['linear', 'conv2d']
        """
        super().__init__(rank_ratio, accuracy_threshold)
        self.layer_types = layer_types or ['linear', 'conv2d']
        
    def svd_decompose_linear(self, linear_layer: nn.Linear, rank: int) -> nn.Sequential:
        """
        Decompose Linear layer using SVD: W ≈ U @ S @ V^T.
        Replace with two Linear layers: Linear(in, rank) -> Linear(rank, out)
        """
        weight = linear_layer.weight.data  # Shape: [out_features, in_features]
        
        # Perform SVD
        U, S, Vt = torch.svd(weight)
        
        # Truncate to desired rank
        rank = min(rank, min(U.size(1), Vt.size(0)))
        U_truncated = U[:, :rank]  # [out_features, rank]
        S_truncated = S[:rank]     # [rank]
        Vt_truncated = Vt[:rank, :]  # [rank, in_features]
        
        # Create two linear layers
        # First layer: input -> rank (V^T with scaling)
        first_layer = nn.Linear(linear_layer.in_features, rank, bias=False)
        first_layer.weight.data = torch.diag(torch.sqrt(S_truncated)) @ Vt_truncated
        
        # Second layer: rank -> output (U with scaling)
        second_layer = nn.Linear(rank, linear_layer.out_features, 
                                bias=linear_layer.bias is not None)
        second_layer.weight.data = U_truncated @ torch.diag(torch.sqrt(S_truncated))
        
        # Copy bias to second layer
        if linear_layer.bias is not None:
            second_layer.bias.data = linear_layer.bias.data
            
        return nn.Sequential(first_layer, second_layer)
    
    def svd_decompose_conv2d(self, conv_layer: nn.Conv2d, rank: int) -> nn.Sequential:
        """
        Decompose Conv2d layer using SVD.
        For simplicity, we decompose the weight tensor by reshaping it to 2D.
        """
        # Get original parameters
        weight = conv_layer.weight.data  # [out_ch, in_ch, H, W]
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        # Reshape weight for SVD: [out_channels, in_channels * H * W]
        weight_2d = weight.view(out_channels, -1)
        
        # Perform SVD
        U, S, Vt = torch.svd(weight_2d)
        
        # Truncate to desired rank
        rank = min(rank, min(U.size(1), Vt.size(0)))
        U_truncated = U[:, :rank]
        S_truncated = S[:rank]
        Vt_truncated = Vt[:rank, :]
        
        # Create two conv layers
        # First conv: input -> rank channels (1x1 conv)
        first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=rank,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=1,  # Can't maintain groups in decomposition
            bias=False
        )
        
        # Reshape Vt back to conv weight format
        first_weight = (torch.diag(torch.sqrt(S_truncated)) @ Vt_truncated).view(
            rank, in_channels, kernel_h, kernel_w
        )
        first_conv.weight.data = first_weight
        
        # Second conv: rank -> output channels (1x1 conv)
        second_conv = nn.Conv2d(
            in_channels=rank,
            out_channels=out_channels,
            kernel_size=1,  # 1x1 conv for channel combination
            stride=1,
            padding=0,
            bias=conv_layer.bias is not None
        )
        
        # Set second conv weight
        second_weight = (U_truncated @ torch.diag(torch.sqrt(S_truncated))).view(
            out_channels, rank, 1, 1
        )
        second_conv.weight.data = second_weight
        
        # Copy bias to second layer
        if conv_layer.bias is not None:
            second_conv.bias.data = conv_layer.bias.data
            
        return nn.Sequential(first_conv, second_conv)
    
    def calculate_layer_rank(self, layer: nn.Module) -> int:
        """Calculate appropriate rank for a layer based on rank_ratio."""
        if isinstance(layer, nn.Linear):
            original_rank = min(layer.in_features, layer.out_features)
        elif isinstance(layer, nn.Conv2d):
            # For conv layers, consider the spatial dimensions as well
            weight_2d_size = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            original_rank = min(layer.out_channels, weight_2d_size)
        else:
            return None
            
        new_rank = max(1, int(original_rank * self.rank_ratio))
        return new_rank
    
    def decompose(self, model: nn.Module, layer_ranks: Optional[Dict[str, int]] = None,
                  **kwargs) -> nn.Module:
        """
        Perform SVD decomposition on the model.
        
        Args:
            model: Original model to decompose
            layer_ranks: Per-layer rank specifications (optional)
            **kwargs: Additional arguments
            
        Returns:
            Model with decomposed layers
        """
        logger.info(f"Starting SVD decomposition with rank ratio {self.rank_ratio:.2f}...")
        
        model_copy = copy.deepcopy(model)
        
        if layer_ranks is None:
            layer_ranks = {}
        
        # Track decomposed layers for replacement
        layers_to_replace = {}
        
        for name, module in model_copy.named_modules():
            decomposed = False
            
            if isinstance(module, nn.Linear) and 'linear' in self.layer_types:
                rank = layer_ranks.get(name, self.calculate_layer_rank(module))
                if rank and rank < min(module.in_features, module.out_features):
                    logger.info(f"Decomposing Linear layer '{name}' to rank {rank}")
                    decomposed_layer = self.svd_decompose_linear(module, rank)
                    layers_to_replace[name] = decomposed_layer
                    decomposed = True
                    
            elif isinstance(module, nn.Conv2d) and 'conv2d' in self.layer_types:
                rank = layer_ranks.get(name, self.calculate_layer_rank(module))
                if rank and rank < module.out_channels:
                    logger.info(f"Decomposing Conv2d layer '{name}' to rank {rank}")
                    decomposed_layer = self.svd_decompose_conv2d(module, rank)
                    layers_to_replace[name] = decomposed_layer
                    decomposed = True
            
            if decomposed:
                logger.info(f"Layer '{name}' decomposed successfully")
        
        # Replace layers in the model
        for layer_name, new_layer in layers_to_replace.items():
            # Navigate to the parent module and replace the layer
            name_parts = layer_name.split('.')
            parent_module = model_copy
            
            # Navigate to parent
            for part in name_parts[:-1]:
                parent_module = getattr(parent_module, part)
            
            # Replace the layer
            setattr(parent_module, name_parts[-1], new_layer)
        
        logger.info(f"SVD decomposition completed. Decomposed {len(layers_to_replace)} layers.")
        return model_copy


class TuckerApproximator(LowRankApproximator):
    """Tucker decomposition for Conv2d layers (experimental)."""
    
    def __init__(self, rank_ratio: float = 0.5, accuracy_threshold: float = 0.02):
        """
        Initialize Tucker approximator.
        
        Args:
            rank_ratio: Ratio of original rank to keep
            accuracy_threshold: Maximum acceptable accuracy drop
        """
        super().__init__(rank_ratio, accuracy_threshold)
        
    def tucker_decompose_conv2d(self, conv_layer: nn.Conv2d, 
                               ranks: Tuple[int, int, int, int]) -> nn.Sequential:
        """
        Perform Tucker decomposition on Conv2d layer.
        This is a simplified implementation - full Tucker would require tensorly.
        """
        # For now, implement a simplified version using SVD on different modes
        logger.warning("Tucker decomposition is experimental and simplified.")
        
        # Fall back to SVD-based approach
        svd_approximator = SVDApproximator(self.rank_ratio, self.accuracy_threshold)
        rank = int(min(conv_layer.out_channels, conv_layer.in_channels) * self.rank_ratio)
        return svd_approximator.svd_decompose_conv2d(conv_layer, rank)
    
    def decompose(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        Perform Tucker decomposition (simplified version).
        
        Args:
            model: Original model to decompose
            **kwargs: Additional arguments
            
        Returns:
            Model with decomposed layers
        """
        logger.info("Starting Tucker decomposition (simplified)...")
        logger.warning("This is a simplified Tucker implementation. "
                      "For full Tucker decomposition, consider using tensorly.")
        
        # Use SVD approximator as fallback
        svd_approximator = SVDApproximator(self.rank_ratio, self.accuracy_threshold)
        return svd_approximator.decompose(model, **kwargs)


# Utility functions
def compress_with_lowrank(model: nn.Module, method: str = 'svd', **kwargs) -> nn.Module:
    """
    Convenience function to compress a model using low-rank approximation.
    
    Args:
        model: Model to compress
        method: Decomposition method ('svd' or 'tucker')
        **kwargs: Arguments for the approximator
        
    Returns:
        Compressed model
    """
    if method.lower() == 'svd':
        approximator = SVDApproximator(**kwargs)
    elif method.lower() == 'tucker':
        approximator = TuckerApproximator(**kwargs)
    else:
        raise ValueError(f"Unknown decomposition method: {method}. Use 'svd' or 'tucker'.")
    
    return approximator.decompose(model, **kwargs)


def evaluate_lowrank_model(original_model: nn.Module, 
                          compressed_model: nn.Module,
                          eval_fn=None,
                          rank_ratio: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate low-rank compression results.
    
    Args:
        original_model: Original uncompressed model
        compressed_model: Low-rank compressed model
        eval_fn: Function to evaluate model accuracy
        rank_ratio: Target rank ratio
        
    Returns:
        Dictionary with evaluation metrics
    """
    approximator = LowRankApproximator(rank_ratio)
    return approximator.evaluate_decomposition(original_model, compressed_model, eval_fn)


def analyze_layer_ranks(model: nn.Module) -> Dict[str, Dict[str, int]]:
    """
    Analyze the rank properties of each layer in the model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with layer rank information
    """
    layer_info = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            info = {
                'type': 'Linear',
                'in_features': module.in_features,
                'out_features': module.out_features,
                'max_rank': min(module.in_features, module.out_features),
                'params': module.in_features * module.out_features
            }
            if module.bias is not None:
                info['params'] += module.out_features
            layer_info[name] = info
            
        elif isinstance(module, nn.Conv2d):
            weight_2d_size = (module.in_channels * 
                            module.kernel_size[0] * 
                            module.kernel_size[1])
            info = {
                'type': 'Conv2d',
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'max_rank': min(module.out_channels, weight_2d_size),
                'params': (module.out_channels * module.in_channels * 
                          module.kernel_size[0] * module.kernel_size[1])
            }
            if module.bias is not None:
                info['params'] += module.out_channels
            layer_info[name] = info
    
    return layer_info


def estimate_compression_savings(model: nn.Module, rank_ratio: float = 0.5) -> Dict[str, Any]:
    """
    Estimate potential compression savings from low-rank approximation.
    
    Args:
        model: Model to analyze
        rank_ratio: Target rank ratio
        
    Returns:
        Dictionary with compression estimates
    """
    layer_info = analyze_layer_ranks(model)
    
    original_params = 0
    compressed_params = 0
    
    for name, info in layer_info.items():
        original_params += info['params']
        
        if info['type'] == 'Linear':
            # SVD decomposition: W[m,n] -> U[m,r] + V[r,n]
            rank = max(1, int(info['max_rank'] * rank_ratio))
            new_params = info['in_features'] * rank + rank * info['out_features']
            compressed_params += new_params
            
        elif info['type'] == 'Conv2d':
            # Simplified conv decomposition estimate
            rank = max(1, int(info['max_rank'] * rank_ratio))
            # First conv: [in_ch, rank, k, k] + Second conv: [rank, out_ch, 1, 1]
            new_params = (info['in_channels'] * rank * 
                         info['kernel_size'][0] * info['kernel_size'][1] +
                         rank * info['out_channels'])
            compressed_params += new_params
    
    compression_ratio = original_params / compressed_params if compressed_params > 0 else float('inf')
    size_reduction = (1 - compressed_params / original_params) * 100 if original_params > 0 else 0
    
    return {
        'original_params': original_params,
        'estimated_compressed_params': compressed_params,
        'compression_ratio': compression_ratio,
        'estimated_size_reduction_percent': size_reduction,
        'original_size_mb': (original_params * 4) / (1024 * 1024),
        'estimated_compressed_size_mb': (compressed_params * 4) / (1024 * 1024)
    }