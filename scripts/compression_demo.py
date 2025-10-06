"""
Demo script for compression techniques.

Demonstrates quantization, pruning, and low-rank approximation on a simple model.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.append('/home/emilio/Documents/ai/moe-img')

from compression.quantize import PTQQuantizer, QATQuantizer
from compression.prune import MagnitudePruner, StructuredPruner
from compression.lowrank import SVDApproximator


class DemoNet(nn.Module):
    """Demo network for compression testing."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model_stats(model: nn.Module) -> dict:
    """Get model statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (total_params * 4) / (1024 * 1024)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': size_mb
    }


def test_model_functionality(model: nn.Module, input_shape=(1, 3, 32, 32)):
    """Test if model works correctly."""
    try:
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(*input_shape)
            output = model(test_input)
            return True, output.shape
    except Exception as e:
        return False, str(e)


def demo_magnitude_pruning():
    """Demonstrate magnitude-based pruning."""
    print("🔪 Magnitude Pruning Demo")
    print("=" * 50)
    
    # Create model
    model = DemoNet()
    original_stats = get_model_stats(model)
    
    print(f"Original model: {original_stats['total_params']:,} params, "
          f"{original_stats['size_mb']:.2f} MB")
    
    # Test original model
    works, result = test_model_functionality(model)
    print(f"Original model functional: {works}, output shape: {result}")
    
    # Apply pruning
    pruner = MagnitudePruner(target_sparsity=0.3, iterative_steps=2)
    pruned_model = pruner.prune(model)
    
    # Get pruned stats
    pruned_stats = get_model_stats(pruned_model)
    sparsity = pruner.get_model_sparsity(pruned_model)
    
    print(f"Pruned model: {pruned_stats['total_params']:,} params, "
          f"{pruned_stats['size_mb']:.2f} MB")
    print(f"Sparsity achieved: {sparsity:.1%}")
    
    # Test pruned model
    works, result = test_model_functionality(pruned_model)
    print(f"Pruned model functional: {works}, output shape: {result}")
    
    # Effective size (considering sparsity)
    effective_size_mb = pruned_stats['size_mb'] * (1 - sparsity)
    print(f"Effective size (sparse): {effective_size_mb:.2f} MB")
    print(f"Size reduction: {(1 - effective_size_mb / original_stats['size_mb']) * 100:.1f}%")
    print()


def demo_structured_pruning():
    """Demonstrate structured pruning."""
    print("🏗️ Structured Pruning Demo")
    print("=" * 50)
    
    # Create model
    model = DemoNet()
    original_stats = get_model_stats(model)
    
    print(f"Original model: {original_stats['total_params']:,} params, "
          f"{original_stats['size_mb']:.2f} MB")
    
    # Apply structured pruning
    pruner = StructuredPruner(target_sparsity=0.25, granularity='filter')
    pruned_model = pruner.prune(model)
    
    # Get pruned stats
    pruned_stats = get_model_stats(model)
    
    print(f"Pruned model: {pruned_stats['total_params']:,} params, "
          f"{pruned_stats['size_mb']:.2f} MB")
    
    # Test pruned model
    works, result = test_model_functionality(pruned_model)
    print(f"Pruned model functional: {works}")
    if works:
        print(f"Output shape: {result}")
    else:
        print(f"Error: {result}")
    
    print(f"Size reduction: {(1 - pruned_stats['size_mb'] / original_stats['size_mb']) * 100:.1f}%")
    print()


def demo_svd_compression():
    """Demonstrate SVD compression."""
    print("📊 SVD Low-Rank Compression Demo")
    print("=" * 50)
    
    # Create model
    model = DemoNet()
    original_stats = get_model_stats(model)
    
    print(f"Original model: {original_stats['total_params']:,} params, "
          f"{original_stats['size_mb']:.2f} MB")
    
    # Apply SVD compression (only linear layers)
    approximator = SVDApproximator(rank_ratio=0.5, layer_types=['linear'])
    compressed_model = approximator.decompose(model)
    
    # Get compressed stats
    compressed_stats = get_model_stats(compressed_model)
    
    print(f"Compressed model: {compressed_stats['total_params']:,} params, "
          f"{compressed_stats['size_mb']:.2f} MB")
    
    # Test compressed model
    works, result = test_model_functionality(compressed_model)
    print(f"Compressed model functional: {works}")
    if works:
        print(f"Output shape: {result}")
    else:
        print(f"Error: {result}")
    
    compression_ratio = original_stats['size_mb'] / compressed_stats['size_mb']
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Size reduction: {(1 - compressed_stats['size_mb'] / original_stats['size_mb']) * 100:.1f}%")
    print()


def demo_quantization():
    """Demonstrate quantization (basic setup)."""
    print("⚡ Quantization Demo (Setup)")
    print("=" * 50)
    
    # Create model
    model = DemoNet()
    original_stats = get_model_stats(model)
    
    print(f"Original model: {original_stats['total_params']:,} params, "
          f"{original_stats['size_mb']:.2f} MB")
    
    # Setup PTQ quantizer
    quantizer = PTQQuantizer(target_size_mb=1.0, bit_width=8)
    
    print("PTQ Quantizer configured:")
    print(f"  Backend: {quantizer.backend}")
    print(f"  Bit width: {quantizer.bit_width}")
    print(f"  Target size: {quantizer.target_size_mb} MB")
    
    # Note: Full quantization would require calibration data
    print("\nNote: Full quantization demo requires calibration data loader")
    print("Theoretical 8-bit quantized size:", original_stats['size_mb'] / 4, "MB")
    print()


def demo_compression_comparison():
    """Compare different compression techniques."""
    print("📈 Compression Techniques Comparison")
    print("=" * 80)
    
    # Original model
    original_model = DemoNet()
    original_stats = get_model_stats(original_model)
    
    results = {
        'Original': {
            'size_mb': original_stats['size_mb'],
            'params': original_stats['total_params'],
            'functional': True
        }
    }
    
    # Magnitude Pruning
    try:
        pruner = MagnitudePruner(target_sparsity=0.3, iterative_steps=1)
        pruned = pruner.prune(DemoNet())
        sparsity = pruner.get_model_sparsity(pruned)
        effective_size = get_model_stats(pruned)['size_mb'] * (1 - sparsity)
        works, _ = test_model_functionality(pruned)
        
        results['Magnitude Pruning (30%)'] = {
            'size_mb': effective_size,
            'params': get_model_stats(pruned)['total_params'],
            'functional': works,
            'sparsity': sparsity
        }
    except Exception as e:
        results['Magnitude Pruning (30%)'] = {'error': str(e)}
    
    # SVD Compression
    try:
        approximator = SVDApproximator(rank_ratio=0.5, layer_types=['linear'])
        compressed = approximator.decompose(DemoNet())
        compressed_stats = get_model_stats(compressed)
        works, _ = test_model_functionality(compressed)
        
        results['SVD Compression (50% rank)'] = {
            'size_mb': compressed_stats['size_mb'],
            'params': compressed_stats['total_params'],
            'functional': works
        }
    except Exception as e:
        results['SVD Compression (50% rank)'] = {'error': str(e)}
    
    # Print comparison table
    print(f"{'Method':<25} {'Size (MB)':<12} {'Parameters':<12} {'Reduction':<12} {'Functional':<12}")
    print("-" * 80)
    
    for method, stats in results.items():
        if 'error' in stats:
            print(f"{method:<25} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'No':<12}")
        else:
            size_reduction = (1 - stats['size_mb'] / original_stats['size_mb']) * 100
            functional = "Yes" if stats['functional'] else "No"
            print(f"{method:<25} {stats['size_mb']:<12.3f} {stats['params']:<12,} "
                  f"{size_reduction:<12.1f}% {functional:<12}")
    
    print()


if __name__ == "__main__":
    print("🧪 Compression Techniques Demo")
    print("=" * 80)
    print()
    
    # Run all demos
    demo_magnitude_pruning()
    demo_structured_pruning()
    demo_svd_compression()
    demo_quantization()
    demo_compression_comparison()
    
    print("✅ Compression demo completed!")