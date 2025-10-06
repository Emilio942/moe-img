"""
Quantization module for model compression.

Provides Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)
to reduce model size while maintaining accuracy.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QConfig, default_observer, default_fake_quant
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import torch.fx
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import copy
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Base class for model quantization."""
    
    def __init__(self, target_size_mb: float = 1.0, accuracy_threshold: float = 0.02):
        """
        Initialize quantizer.
        
        Args:
            target_size_mb: Target model size in MB
            accuracy_threshold: Maximum acceptable accuracy drop (e.g., 0.02 = 2%)
        """
        self.target_size_mb = target_size_mb
        self.accuracy_threshold = accuracy_threshold
        
    def get_model_size_mb(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb
    
    def quantize(self, model: nn.Module, **kwargs) -> nn.Module:
        """Abstract method for quantization."""
        raise NotImplementedError
        
    def evaluate_compression(self, original_model: nn.Module, 
                           quantized_model: nn.Module,
                           eval_fn) -> Dict[str, Any]:
        """Evaluate compression results."""
        original_size = self.get_model_size_mb(original_model)
        quantized_size = self.get_model_size_mb(quantized_model)
        compression_ratio = original_size / quantized_size
        
        # Evaluate accuracy (if eval function provided)
        original_acc = eval_fn(original_model) if eval_fn else None
        quantized_acc = eval_fn(quantized_model) if eval_fn else None
        accuracy_drop = (original_acc - quantized_acc) if (original_acc and quantized_acc) else None
        
        return {
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': compression_ratio,
            'size_reduction_percent': (1 - quantized_size/original_size) * 100,
            'original_accuracy': original_acc,
            'quantized_accuracy': quantized_acc,
            'accuracy_drop': accuracy_drop,
            'meets_size_target': quantized_size <= self.target_size_mb,
            'meets_accuracy_target': accuracy_drop <= self.accuracy_threshold if accuracy_drop else None
        }


class PTQQuantizer(ModelQuantizer):
    """Post-Training Quantization (PTQ) implementation."""
    
    def __init__(self, target_size_mb: float = 1.0, accuracy_threshold: float = 0.02,
                 backend: str = 'fbgemm', bit_width: int = 8):
        """
        Initialize PTQ quantizer.
        
        Args:
            target_size_mb: Target model size in MB
            accuracy_threshold: Maximum acceptable accuracy drop
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
            bit_width: Quantization bit width (8 or 4)
        """
        super().__init__(target_size_mb, accuracy_threshold)
        self.backend = backend
        self.bit_width = bit_width
        
        # Set quantization backend
        torch.backends.quantized.engine = backend
        
    def prepare_model_for_quantization(self, model: nn.Module) -> nn.Module:
        """Prepare model for quantization by adding observers."""
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Set quantization config
        if self.bit_width == 8:
            qconfig = torch.quantization.get_default_qconfig(self.backend)
        else:  # 4-bit or custom
            # Use reduced precision observers for 4-bit
            qconfig = QConfig(
                activation=default_observer.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine),
                weight=default_observer.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
            )
        
        model_copy.qconfig = qconfig
        
        # Prepare model (adds observers)
        try:
            # Try FX graph mode quantization first
            example_input = torch.randn(1, 3, 32, 32)  # CIFAR-10 input shape
            prepared = prepare_fx(model_copy, {"": qconfig}, example_inputs=example_input)
        except Exception as e:
            logger.warning(f"FX quantization failed: {e}. Falling back to eager mode.")
            # Fall back to eager mode quantization
            prepared = torch.quantization.prepare(model_copy, inplace=False)
            
        return prepared
        
    def calibrate_model(self, prepared_model: nn.Module, 
                       calibration_loader, num_batches: int = 100) -> nn.Module:
        """Calibrate model with representative data."""
        logger.info(f"Calibrating model with {num_batches} batches...")
        
        prepared_model.eval()
        with torch.no_grad():
            for i, (data, _) in enumerate(calibration_loader):
                if i >= num_batches:
                    break
                    
                try:
                    _ = prepared_model(data)
                except Exception as e:
                    logger.warning(f"Calibration batch {i} failed: {e}")
                    continue
                    
        logger.info("Calibration completed.")
        return prepared_model
        
    def quantize(self, model: nn.Module, calibration_loader=None, 
                 num_calibration_batches: int = 100) -> nn.Module:
        """
        Perform Post-Training Quantization.
        
        Args:
            model: Original model to quantize
            calibration_loader: DataLoader for calibration (optional)
            num_calibration_batches: Number of batches for calibration
            
        Returns:
            Quantized model
        """
        logger.info("Starting Post-Training Quantization...")
        
        # Prepare model for quantization
        prepared_model = self.prepare_model_for_quantization(model)
        
        # Calibrate if calibration data provided
        if calibration_loader is not None:
            prepared_model = self.calibrate_model(
                prepared_model, calibration_loader, num_calibration_batches
            )
        
        # Convert to quantized model
        try:
            # Try FX conversion
            quantized_model = convert_fx(prepared_model)
        except Exception as e:
            logger.warning(f"FX conversion failed: {e}. Falling back to eager mode.")
            # Fall back to eager mode conversion
            quantized_model = torch.quantization.convert(prepared_model, inplace=False)
            
        logger.info("PTQ completed successfully.")
        return quantized_model


class QATQuantizer(ModelQuantizer):
    """Quantization-Aware Training (QAT) implementation."""
    
    def __init__(self, target_size_mb: float = 1.0, accuracy_threshold: float = 0.01,
                 backend: str = 'fbgemm', num_epochs: int = 5):
        """
        Initialize QAT quantizer.
        
        Args:
            target_size_mb: Target model size in MB
            accuracy_threshold: Maximum acceptable accuracy drop
            backend: Quantization backend
            num_epochs: Number of fine-tuning epochs
        """
        super().__init__(target_size_mb, accuracy_threshold)
        self.backend = backend
        self.num_epochs = num_epochs
        
        # Set quantization backend
        torch.backends.quantized.engine = backend
        
    def prepare_qat_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for QAT by adding fake quantization."""
        model_copy = copy.deepcopy(model)
        model_copy.train()
        
        # Set QAT config with fake quantization
        qat_qconfig = torch.quantization.get_default_qat_qconfig(self.backend)
        model_copy.qconfig = qat_qconfig
        
        try:
            # Try FX graph mode QAT
            example_input = torch.randn(1, 3, 32, 32)
            prepared = prepare_fx(model_copy, {"": qat_qconfig}, example_inputs=example_input)
        except Exception as e:
            logger.warning(f"FX QAT preparation failed: {e}. Falling back to eager mode.")
            # Fall back to eager mode QAT
            prepared = torch.quantization.prepare_qat(model_copy, inplace=False)
            
        return prepared
        
    def fine_tune_model(self, prepared_model: nn.Module, 
                       train_loader, optimizer, criterion,
                       device: str = 'cpu') -> nn.Module:
        """Fine-tune model with fake quantization."""
        logger.info(f"Fine-tuning model for {self.num_epochs} epochs with QAT...")
        
        prepared_model.to(device)
        prepared_model.train()
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = prepared_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Log progress every 50 batches
                if batch_idx % 50 == 0:
                    logger.info(f'Epoch {epoch+1}/{self.num_epochs}, '
                              f'Batch {batch_idx}, Loss: {loss.item():.4f}')
                    
            avg_loss = epoch_loss / num_batches
            logger.info(f'Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}')
            
        logger.info("QAT fine-tuning completed.")
        return prepared_model
        
    def quantize(self, model: nn.Module, train_loader, 
                 optimizer=None, criterion=None, device: str = 'cpu') -> nn.Module:
        """
        Perform Quantization-Aware Training.
        
        Args:
            model: Original model to quantize
            train_loader: Training data loader
            optimizer: Optimizer for fine-tuning (optional, will create AdamW if None)
            criterion: Loss function (optional, will create CrossEntropyLoss if None)
            device: Device to train on
            
        Returns:
            Quantized model
        """
        logger.info("Starting Quantization-Aware Training...")
        
        # Prepare model for QAT
        prepared_model = self.prepare_qat_model(model)
        
        # Setup optimizer and criterion if not provided
        if optimizer is None:
            optimizer = torch.optim.AdamW(prepared_model.parameters(), lr=1e-4, weight_decay=1e-4)
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        # Fine-tune with fake quantization
        fine_tuned_model = self.fine_tune_model(
            prepared_model, train_loader, optimizer, criterion, device
        )
        
        # Convert to actual quantized model
        fine_tuned_model.eval()
        try:
            quantized_model = convert_fx(fine_tuned_model)
        except Exception as e:
            logger.warning(f"FX conversion failed: {e}. Falling back to eager mode.")
            quantized_model = torch.quantization.convert(fine_tuned_model.eval(), inplace=False)
            
        logger.info("QAT completed successfully.")
        return quantized_model


# Utility functions
def quantize_model(model: nn.Module, method: str = 'ptq', **kwargs) -> nn.Module:
    """
    Convenience function to quantize a model.
    
    Args:
        model: Model to quantize
        method: Quantization method ('ptq' or 'qat')
        **kwargs: Arguments for the quantizer
        
    Returns:
        Quantized model
    """
    if method.lower() == 'ptq':
        quantizer = PTQQuantizer(**kwargs)
        return quantizer.quantize(model, **kwargs)
    elif method.lower() == 'qat':
        quantizer = QATQuantizer(**kwargs)
        return quantizer.quantize(model, **kwargs)
    else:
        raise ValueError(f"Unknown quantization method: {method}. Use 'ptq' or 'qat'.")


def evaluate_quantized_model(original_model: nn.Module, 
                           quantized_model: nn.Module,
                           eval_fn=None,
                           target_size_mb: float = 1.0) -> Dict[str, Any]:
    """
    Evaluate quantization results.
    
    Args:
        original_model: Original unquantized model
        quantized_model: Quantized model
        eval_fn: Function to evaluate model accuracy
        target_size_mb: Target size threshold
        
    Returns:
        Dictionary with evaluation metrics
    """
    quantizer = ModelQuantizer(target_size_mb)
    return quantizer.evaluate_compression(original_model, quantized_model, eval_fn)


def benchmark_quantized_inference(model: nn.Module, 
                                input_shape: Tuple[int, ...] = (1, 3, 32, 32),
                                num_runs: int = 100,
                                warmup_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark inference time of quantized model.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Timing statistics
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    import numpy as np
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99)
    }