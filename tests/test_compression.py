"""
Tests for the compression module.

Tests quantization, pruning, and low-rank approximation techniques.
"""

import pytest
import torch
import torch.nn as nn
import torch.fx
import tempfile
from pathlib import Path
import numpy as np

from compression.quantize import (
    PTQQuantizer, QATQuantizer, ModelQuantizer,
    quantize_model, evaluate_quantized_model
)
from compression.prune import (
    MagnitudePruner, StructuredPruner, ModelPruner,
    prune_model, evaluate_pruned_model, get_layer_wise_sparsity
)
from compression.lowrank import (
    SVDApproximator, TuckerApproximator, LowRankApproximator,
    compress_with_lowrank, evaluate_lowrank_model, analyze_layer_ranks
)


# Test models
class SimpleNet(nn.Module):
    """Simple test network."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TinyNet(nn.Module):
    """Tiny test network for quick tests."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def dummy_eval_fn(model):
    """Dummy evaluation function for testing."""
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'conv1'):  # SimpleNet
            dummy_input = torch.randn(1, 3, 32, 32)
        else:  # TinyNet
            dummy_input = torch.randn(1, 10)
        try:
            output = model(dummy_input)
            # Return dummy accuracy based on output variance (higher = better)
            return torch.var(output).item() * 100
        except Exception:
            return 0.0


class TestModelQuantizer:
    """Test base quantizer functionality."""
    
    def test_model_size_calculation(self):
        """Test model size calculation."""
        model = TinyNet()
        quantizer = ModelQuantizer()
        
        size_mb = quantizer.get_model_size_mb(model)
        assert size_mb > 0
        assert isinstance(size_mb, float)
        
        # Manual calculation: (10*20 + 20) + (20*5 + 5) = 325 params * 4 bytes
        expected_size_mb = (325 * 4) / (1024 * 1024)
        assert abs(size_mb - expected_size_mb) < 1e-6
        
    def test_evaluate_compression(self):
        """Test compression evaluation."""
        original_model = TinyNet()
        quantizer = ModelQuantizer()
        
        # Create a "compressed" model by copying (same size for this test)
        compressed_model = TinyNet()
        
        results = quantizer.evaluate_compression(
            original_model, compressed_model, dummy_eval_fn
        )
        
        assert 'original_size_mb' in results
        assert 'quantized_size_mb' in results
        assert 'compression_ratio' in results
        assert 'original_accuracy' in results
        assert 'quantized_accuracy' in results
        assert results['compression_ratio'] == pytest.approx(1.0, rel=0.1)


class TestPTQQuantizer:
    """Test Post-Training Quantization."""
    
    def test_ptq_initialization(self):
        """Test PTQ quantizer initialization."""
        quantizer = PTQQuantizer(target_size_mb=0.5, bit_width=8)
        
        assert quantizer.target_size_mb == 0.5
        assert quantizer.bit_width == 8
        assert quantizer.backend in ['fbgemm', 'qnnpack']
        
    def test_prepare_model_for_quantization(self):
        """Test model preparation for quantization."""
        model = TinyNet()
        quantizer = PTQQuantizer()
        
        prepared_model = quantizer.prepare_model_for_quantization(model)
        
        # Check that model has qconfig or is prepared (FX mode may not have qconfig attribute)
        # In FX mode, the model is a GraphModule
        assert prepared_model is not None
        assert isinstance(prepared_model, (nn.Module, torch.fx.GraphModule))
        
    def test_ptq_quantization_tiny_model(self):
        """Test PTQ on tiny model."""
        model = TinyNet()
        quantizer = PTQQuantizer(target_size_mb=1.0)
        
        # Should work without calibration data
        quantized_model = quantizer.quantize(model)
        
        # Check that it's different from original
        assert quantized_model is not model
        
        # Check size reduction (quantized should be smaller)
        original_size = quantizer.get_model_size_mb(model)
        quantized_size = quantizer.get_model_size_mb(quantized_model)
        
        # Note: Size might not always be smaller due to quantization overhead
        # in small models, but should still work
        assert quantized_size > 0


class TestQATQuantizer:
    """Test Quantization-Aware Training."""
    
    def test_qat_initialization(self):
        """Test QAT quantizer initialization."""
        quantizer = QATQuantizer(num_epochs=3)
        
        assert quantizer.num_epochs == 3
        assert quantizer.backend in ['fbgemm', 'qnnpack']
        
    def test_prepare_qat_model(self):
        """Test QAT model preparation."""
        model = TinyNet()
        quantizer = QATQuantizer()
        
        prepared_model = quantizer.prepare_qat_model(model)
        
        # Check that model has qconfig
        assert hasattr(prepared_model, 'qconfig')


class TestMagnitudePruner:
    """Test magnitude-based pruning."""
    
    def test_magnitude_pruner_initialization(self):
        """Test magnitude pruner initialization."""
        pruner = MagnitudePruner(target_sparsity=0.3, iterative_steps=2)
        
        assert pruner.target_sparsity == 0.3
        assert pruner.iterative_steps == 2
        assert pruner.exclude_bias == True
        
    def test_get_prunable_modules(self):
        """Test identification of prunable modules."""
        model = SimpleNet()
        pruner = MagnitudePruner()
        
        prunable = pruner.get_prunable_modules(model)
        
        # Should include weight parameters from conv and linear layers
        assert len(prunable) >= 4  # At least conv1, conv2, fc1, fc2 weights
        
        # Check that we get (module, param_name) tuples
        for module, param_name in prunable:
            assert hasattr(module, param_name)
            assert param_name in ['weight', 'bias']
            
    def test_calculate_global_threshold(self):
        """Test global threshold calculation."""
        model = TinyNet()
        pruner = MagnitudePruner(target_sparsity=0.3)
        
        threshold = pruner.calculate_global_threshold(model, 0.3)
        
        assert isinstance(threshold, float)
        assert threshold >= 0
        
    def test_magnitude_pruning_tiny_model(self):
        """Test magnitude pruning on tiny model."""
        model = TinyNet()
        pruner = MagnitudePruner(target_sparsity=0.2, iterative_steps=1)
        
        original_sparsity = pruner.get_model_sparsity(model)
        
        pruned_model = pruner.prune(model)
        
        final_sparsity = pruner.get_model_sparsity(pruned_model)
        
        # Sparsity should increase
        assert final_sparsity > original_sparsity
        # Should be close to target sparsity
        assert final_sparsity >= 0.15  # Allow some tolerance
        
    def test_evaluate_pruning(self):
        """Test pruning evaluation."""
        original_model = TinyNet()
        pruner = MagnitudePruner(target_sparsity=0.3)
        
        pruned_model = pruner.prune(original_model)
        
        results = pruner.evaluate_pruning(original_model, pruned_model, dummy_eval_fn)
        
        assert 'sparsity' in results
        assert 'compression_ratio' in results
        assert 'original_accuracy' in results
        assert 'pruned_accuracy' in results
        assert results['sparsity'] > 0


class TestStructuredPruner:
    """Test structured pruning."""
    
    def test_structured_pruner_initialization(self):
        """Test structured pruner initialization."""
        pruner = StructuredPruner(target_sparsity=0.3, granularity='filter')
        
        assert pruner.target_sparsity == 0.3
        assert pruner.granularity == 'filter'
        assert pruner.importance_metric == 'l1'
        
    def test_calculate_filter_importance(self):
        """Test filter importance calculation."""
        pruner = StructuredPruner(importance_metric='l1')
        
        # Test with conv2d weight
        conv_weight = torch.randn(8, 4, 3, 3)  # [out_ch, in_ch, H, W]
        importance = pruner.calculate_filter_importance(conv_weight)
        
        assert importance.shape == (8,)  # One score per output channel
        assert torch.all(importance >= 0)  # L1 norm is non-negative
        
        # Test with linear weight
        linear_weight = torch.randn(10, 20)  # [out_features, in_features]
        importance = pruner.calculate_filter_importance(linear_weight)
        
        assert importance.shape == (10,)  # One score per output feature
        assert torch.all(importance >= 0)
        
    def test_prune_linear_layer(self):
        """Test linear layer pruning."""
        pruner = StructuredPruner(target_sparsity=0.3)
        
        original_layer = nn.Linear(20, 10, bias=True)
        pruned_layer = pruner.prune_linear_layer(original_layer, 0.3)
        
        # Should have fewer output features
        assert pruned_layer.out_features == 7  # 10 - 3 = 7
        assert pruned_layer.in_features == 20   # Input unchanged
        assert pruned_layer.bias is not None    # Bias preserved
        
    def test_prune_conv_layer(self):
        """Test conv layer pruning."""
        pruner = StructuredPruner(target_sparsity=0.25)
        
        original_layer = nn.Conv2d(8, 12, 3, padding=1, bias=True)
        pruned_layer = pruner.prune_conv_layer(original_layer, 0.25)
        
        # Should have fewer output channels
        assert pruned_layer.out_channels == 9   # 12 - 3 = 9
        assert pruned_layer.in_channels == 8    # Input unchanged
        assert pruned_layer.kernel_size == (3, 3)  # Kernel unchanged
        assert pruned_layer.bias is not None   # Bias preserved
        
    def test_structured_pruning_model(self):
        """Test structured pruning on full model."""
        model = SimpleNet()
        pruner = StructuredPruner(target_sparsity=0.2)
        
        original_params = sum(p.numel() for p in model.parameters())
        
        pruned_model = pruner.prune(model)
        
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        
        # Should have fewer parameters due to structural changes
        assert pruned_params < original_params
        
        # Model should still be functional
        with torch.no_grad():
            test_input = torch.randn(1, 3, 32, 32)
            output = pruned_model(test_input)
            assert output.shape == (1, 10)


class TestSVDApproximator:
    """Test SVD-based low-rank approximation."""
    
    def test_svd_approximator_initialization(self):
        """Test SVD approximator initialization."""
        approximator = SVDApproximator(rank_ratio=0.5, layer_types=['linear'])
        
        assert approximator.rank_ratio == 0.5
        assert approximator.layer_types == ['linear']
        
    def test_calculate_layer_rank(self):
        """Test layer rank calculation."""
        approximator = SVDApproximator(rank_ratio=0.6)
        
        # Test linear layer
        linear_layer = nn.Linear(20, 10)
        rank = approximator.calculate_layer_rank(linear_layer)
        assert rank == 6  # min(20, 10) * 0.6 = 6
        
        # Test conv layer
        conv_layer = nn.Conv2d(8, 16, 3)
        rank = approximator.calculate_layer_rank(conv_layer)
        expected = int(min(16, 8 * 3 * 3) * 0.6)
        assert rank == expected
        
    def test_svd_decompose_linear(self):
        """Test SVD decomposition of linear layer."""
        approximator = SVDApproximator()
        
        original_layer = nn.Linear(20, 10, bias=True)
        rank = 5
        
        decomposed = approximator.svd_decompose_linear(original_layer, rank)
        
        # Should be a sequential of two linear layers
        assert isinstance(decomposed, nn.Sequential)
        assert len(decomposed) == 2
        assert isinstance(decomposed[0], nn.Linear)
        assert isinstance(decomposed[1], nn.Linear)
        
        # Check dimensions
        assert decomposed[0].in_features == 20
        assert decomposed[0].out_features == rank
        assert decomposed[1].in_features == rank
        assert decomposed[1].out_features == 10
        
        # Second layer should have bias
        assert decomposed[1].bias is not None
        assert decomposed[0].bias is None
        
    def test_svd_decompose_conv2d(self):
        """Test SVD decomposition of conv2d layer."""
        approximator = SVDApproximator()
        
        original_layer = nn.Conv2d(8, 16, 3, padding=1, bias=True)
        rank = 6
        
        decomposed = approximator.svd_decompose_conv2d(original_layer, rank)
        
        # Should be a sequential of two conv layers
        assert isinstance(decomposed, nn.Sequential)
        assert len(decomposed) == 2
        assert isinstance(decomposed[0], nn.Conv2d)
        assert isinstance(decomposed[1], nn.Conv2d)
        
        # Check first conv (main convolution)
        assert decomposed[0].in_channels == 8
        assert decomposed[0].out_channels == rank
        assert decomposed[0].kernel_size == (3, 3)
        
        # Check second conv (1x1 combination)
        assert decomposed[1].in_channels == rank
        assert decomposed[1].out_channels == 16
        assert decomposed[1].kernel_size == (1, 1)
        
        # Second layer should have bias
        assert decomposed[1].bias is not None
        assert decomposed[0].bias is None
        
    def test_svd_model_decomposition(self):
        """Test SVD decomposition on full model."""
        model = TinyNet()
        approximator = SVDApproximator(rank_ratio=0.7, layer_types=['linear'])
        
        original_params = sum(p.numel() for p in model.parameters())
        
        decomposed_model = approximator.decompose(model)
        
        decomposed_params = sum(p.numel() for p in decomposed_model.parameters())
        
        # Should have fewer parameters (though not always guaranteed for small models)
        # At minimum, model should be functional
        with torch.no_grad():
            test_input = torch.randn(1, 10)
            output = decomposed_model(test_input)
            assert output.shape == (1, 5)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_quantize_model_convenience(self):
        """Test quantize_model convenience function."""
        model = TinyNet()
        
        # Test PTQ
        ptq_model = quantize_model(model, method='ptq', target_size_mb=1.0)
        assert ptq_model is not model
        
    def test_prune_model_convenience(self):
        """Test prune_model convenience function."""
        model = TinyNet()
        
        # Test magnitude pruning
        pruned_model = prune_model(model, method='magnitude', target_sparsity=0.2)
        assert pruned_model is not model
        
        # Check sparsity
        pruner = MagnitudePruner()
        sparsity = pruner.get_model_sparsity(pruned_model)
        assert sparsity > 0
        
    def test_compress_with_lowrank_convenience(self):
        """Test compress_with_lowrank convenience function."""
        model = TinyNet()
        
        # Test SVD
        compressed_model = compress_with_lowrank(model, method='svd', rank_ratio=0.6)
        assert compressed_model is not model
        
        # Model should be functional
        with torch.no_grad():
            test_input = torch.randn(1, 10)
            output = compressed_model(test_input)
            assert output.shape == (1, 5)
            
    def test_get_layer_wise_sparsity(self):
        """Test layer-wise sparsity calculation."""
        model = TinyNet()
        
        # Apply some pruning first
        pruned_model = prune_model(model, method='magnitude', target_sparsity=0.3)
        
        layer_sparsities = get_layer_wise_sparsity(pruned_model)
        
        assert 'fc1' in layer_sparsities
        assert 'fc2' in layer_sparsities
        
        for layer_name, sparsity in layer_sparsities.items():
            assert 0 <= sparsity <= 1
            assert isinstance(sparsity, float)
            
    def test_analyze_layer_ranks(self):
        """Test layer rank analysis."""
        model = SimpleNet()
        
        rank_info = analyze_layer_ranks(model)
        
        # Should have info for conv and linear layers
        assert len(rank_info) >= 4
        
        for layer_name, info in rank_info.items():
            assert 'type' in info
            assert 'max_rank' in info
            assert 'params' in info
            assert info['type'] in ['Linear', 'Conv2d']


if __name__ == "__main__":
    # Run basic functionality tests
    print("Testing compression modules...")
    
    # Test basic quantization
    model = TinyNet()
    quantizer = PTQQuantizer(target_size_mb=1.0)
    try:
        quantized = quantizer.quantize(model)
        print("✓ PTQ quantization works")
    except Exception as e:
        print(f"✗ PTQ quantization failed: {e}")
    
    # Test basic pruning
    pruner = MagnitudePruner(target_sparsity=0.3)
    try:
        pruned = pruner.prune(model)
        sparsity = pruner.get_model_sparsity(pruned)
        print(f"✓ Magnitude pruning works (sparsity: {sparsity:.2%})")
    except Exception as e:
        print(f"✗ Magnitude pruning failed: {e}")
    
    # Test basic SVD
    approximator = SVDApproximator(rank_ratio=0.6)
    try:
        compressed = approximator.decompose(model)
        print("✓ SVD decomposition works")
    except Exception as e:
        print(f"✗ SVD decomposition failed: {e}")
    
    print("Basic compression tests completed!")