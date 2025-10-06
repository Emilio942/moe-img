"""
Simple compression demo that shows basic functionality.
"""

import torch
import torch.nn as nn
import sys

# Add project root to path
sys.path.append('/home/emilio/Documents/ai/moe-img')

from compression.quantize import PTQQuantizer
from compression.prune import MagnitudePruner
from compression.lowrank import SVDApproximator


class SimpleModel(nn.Module):
    """Simple model for compression testing."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model_size_mb(model):
    """Get model size in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    return (total_params * 4) / (1024 * 1024)


def test_compression_techniques():
    """Test all compression techniques."""
    
    print("🧪 Simple Compression Demo")
    print("=" * 50)
    
    # Original model
    model = SimpleModel()
    original_size = get_model_size_mb(model)
    
    print(f"📊 Original Model:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Size: {original_size:.4f} MB")
    
    # Test functionality
    with torch.no_grad():
        test_input = torch.randn(5, 100)
        output = model(test_input)
        print(f"   Output shape: {output.shape}")
        print(f"   ✅ Model is functional")
    
    print()
    
    # 1. Magnitude Pruning
    print("🔪 Testing Magnitude Pruning...")
    try:
        pruner = MagnitudePruner(target_sparsity=0.3, iterative_steps=1)
        pruned_model = pruner.prune(SimpleModel())  # Use fresh model
        
        sparsity = pruner.get_model_sparsity(pruned_model)
        effective_size = get_model_size_mb(pruned_model) * (1 - sparsity)
        
        print(f"   Sparsity: {sparsity:.1%}")
        print(f"   Effective size: {effective_size:.4f} MB")
        print(f"   Reduction: {(1 - effective_size/original_size)*100:.1f}%")
        
        # Test functionality
        with torch.no_grad():
            output = pruned_model(test_input)
            print(f"   ✅ Pruned model works, output: {output.shape}")
            
    except Exception as e:
        print(f"   ❌ Pruning failed: {e}")
    
    print()
    
    # 2. SVD Compression
    print("📊 Testing SVD Compression...")
    try:
        approximator = SVDApproximator(rank_ratio=0.6, layer_types=['linear'])
        compressed_model = approximator.decompose(SimpleModel())  # Use fresh model
        
        compressed_size = get_model_size_mb(compressed_model)
        compression_ratio = original_size / compressed_size
        
        print(f"   Compressed size: {compressed_size:.4f} MB")
        print(f"   Compression ratio: {compression_ratio:.2f}x")
        print(f"   Reduction: {(1 - compressed_size/original_size)*100:.1f}%")
        
        # Test functionality
        with torch.no_grad():
            output = compressed_model(test_input)
            print(f"   ✅ Compressed model works, output: {output.shape}")
            
    except Exception as e:
        print(f"   ❌ SVD compression failed: {e}")
    
    print()
    
    # 3. Quantization Setup
    print("⚡ Testing Quantization Setup...")
    try:
        quantizer = PTQQuantizer(target_size_mb=0.1, bit_width=8)
        
        print(f"   Backend: {quantizer.backend}")
        print(f"   Target size: {quantizer.target_size_mb} MB")
        print(f"   Theoretical 8-bit size: {original_size/4:.4f} MB")
        print("   ✅ Quantizer configured (full quantization needs calibration data)")
        
    except Exception as e:
        print(f"   ❌ Quantization setup failed: {e}")
    
    print()
    print("🎉 Compression demo completed!")


if __name__ == "__main__":
    test_compression_techniques()