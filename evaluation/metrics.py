"""
Core Metrics Collection for Adaptive Evaluation Suite
====================================================

Comprehensive metrics for accuracy, cost (time/memory/energy), robustness,
and calibration. Designed for reproducible and fair model comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from contextlib import contextmanager
import json
import subprocess
import platform

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Container for evaluation metrics with metadata."""
    
    # Core accuracy metrics
    top1_accuracy: float
    top5_accuracy: float
    loss: float
    
    # Cost metrics
    time_per_batch_median: float
    time_per_batch_p95: float
    memory_peak_mb: float
    energy_proxy: float
    
    # Model size metrics
    param_count: int
    model_size_mb: float
    
    # Robustness metrics (optional)
    confidence_mean: Optional[float] = None
    entropy_mean: Optional[float] = None
    
    # Calibration metrics (optional)
    ece: Optional[float] = None
    mce: Optional[float] = None
    
    # Metadata
    num_samples: int = 0
    batch_size: int = 0
    num_batches: int = 0
    seed: int = 42
    
    # Additional info
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoreMetrics:
    """
    Core accuracy and loss metrics with efficient computation.
    """
    
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        """Reset metric accumulators."""
        self.correct_top1 = 0
        self.correct_top5 = 0
        self.total_samples = 0
        self.total_loss = 0.0
        self.num_batches = 0
        
        # For confidence and entropy
        self.confidences = []
        self.entropies = []
        
    def update(self, 
               outputs: torch.Tensor, 
               targets: torch.Tensor,
               loss: Optional[torch.Tensor] = None):
        """
        Update metrics with batch results.
        
        Args:
            outputs: Model logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            loss: Optional loss tensor
        """
        batch_size = targets.size(0)
        
        # Top-1 and Top-5 accuracy
        _, pred = outputs.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        self.correct_top1 += correct[:1].reshape(-1).float().sum(0).item()
        self.correct_top5 += correct[:5].reshape(-1).float().sum(0).item()
        self.total_samples += batch_size
        
        # Loss accumulation
        if loss is not None:
            self.total_loss += loss.item() * batch_size
            
        self.num_batches += 1
        
        # Confidence and entropy (for robustness analysis)
        with torch.no_grad():
            probs = F.softmax(outputs, dim=1)
            
            # Confidence (max probability)
            confidences = probs.max(dim=1)[0]
            self.confidences.extend(confidences.cpu().numpy())
            
            # Entropy
            entropies = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            self.entropies.extend(entropies.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        if self.total_samples == 0:
            return {
                'top1_accuracy': 0.0,
                'top5_accuracy': 0.0,
                'loss': float('inf'),
                'confidence_mean': 0.0,
                'entropy_mean': 0.0
            }
            
        metrics = {
            'top1_accuracy': 100.0 * self.correct_top1 / self.total_samples,
            'top5_accuracy': 100.0 * self.correct_top5 / self.total_samples,
            'loss': self.total_loss / self.total_samples,
            'confidence_mean': np.mean(self.confidences) if self.confidences else 0.0,
            'entropy_mean': np.mean(self.entropies) if self.entropies else 0.0
        }
        
        return metrics


class CostMetrics:
    """
    Cost metrics for time, memory, and energy consumption.
    Includes warmup handling and outlier filtering.
    """
    
    def __init__(self, warmup_batches: int = 2):
        self.warmup_batches = warmup_batches
        self.reset()
        
    def reset(self):
        """Reset cost accumulators."""
        self.batch_times = []
        self.memory_usage = []
        self.batch_count = 0
        
        # GPU memory tracking
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            torch.cuda.reset_peak_memory_stats()
            
    @contextmanager
    def measure_batch(self):
        """Context manager to measure single batch cost."""
        # Memory before batch
        gc.collect()
        if self.gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        memory_before = self._get_memory_usage()
        
        # Start timing
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            # End timing
            if self.gpu_available:
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Memory after batch
            memory_after = self._get_memory_usage()
            
            # Record measurements (skip warmup)
            batch_time = end_time - start_time
            if self.batch_count >= self.warmup_batches:
                self.batch_times.append(batch_time)
                self.memory_usage.append(memory_after)
                
            self.batch_count += 1
            
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.gpu_available:
            # GPU memory
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            # CPU memory
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
            
    def compute(self) -> Dict[str, float]:
        """Compute cost metrics with outlier handling."""
        if not self.batch_times:
            return {
                'time_per_batch_median': 0.0,
                'time_per_batch_p95': 0.0,
                'memory_peak_mb': 0.0,
                'energy_proxy': 0.0
            }
            
        # Time metrics with outlier filtering
        times_array = np.array(self.batch_times)
        
        # Remove extreme outliers (beyond 99th percentile)
        p99 = np.percentile(times_array, 99)
        filtered_times = times_array[times_array <= p99]
        
        time_median = np.median(filtered_times)
        time_p95 = np.percentile(filtered_times, 95)
        
        # Memory peak
        memory_peak = np.max(self.memory_usage) if self.memory_usage else 0.0
        
        # Energy proxy (simplified model: time * compute intensity)
        # This is a rough approximation for comparison purposes
        energy_proxy = time_median * self._get_compute_intensity()
        
        return {
            'time_per_batch_median': time_median,
            'time_per_batch_p95': time_p95,
            'memory_peak_mb': memory_peak,
            'energy_proxy': energy_proxy
        }
        
    def _get_compute_intensity(self) -> float:
        """Get compute intensity factor for energy proxy."""
        if self.gpu_available:
            # GPU-based intensity (higher)
            return 2.0
        else:
            # CPU-based intensity (lower)
            return 1.0


class RobustnessMetrics:
    """
    Robustness metrics for OOD evaluation and corruption resistance.
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset robustness accumulators."""
        self.accuracy_by_corruption = {}
        self.confidence_by_corruption = {}
        self.in_domain_accuracy = None
        
    def update_in_domain(self, accuracy: float):
        """Set in-domain accuracy baseline."""
        self.in_domain_accuracy = accuracy
        
    def update_corruption(self, 
                         corruption_name: str,
                         accuracy: float,
                         confidence: float):
        """Update metrics for specific corruption."""
        self.accuracy_by_corruption[corruption_name] = accuracy
        self.confidence_by_corruption[corruption_name] = confidence
        
    def compute_robustness_scores(self) -> Dict[str, float]:
        """Compute robustness summary metrics."""
        if not self.accuracy_by_corruption or self.in_domain_accuracy is None:
            return {
                'relative_robustness_mean': 0.0,
                'relative_robustness_std': 0.0,
                'accuracy_drop_mean': 0.0,
                'confidence_robustness_mean': 0.0
            }
            
        # Relative robustness (OOD_acc / ID_acc)
        relative_robustness = []
        accuracy_drops = []
        
        for corruption_name, ood_acc in self.accuracy_by_corruption.items():
            rel_robust = ood_acc / max(self.in_domain_accuracy, 1e-10)
            acc_drop = self.in_domain_accuracy - ood_acc
            
            relative_robustness.append(rel_robust)
            accuracy_drops.append(acc_drop)
            
        # Confidence robustness
        confidence_values = list(self.confidence_by_corruption.values())
        
        return {
            'relative_robustness_mean': np.mean(relative_robustness),
            'relative_robustness_std': np.std(relative_robustness),
            'accuracy_drop_mean': np.mean(accuracy_drops),
            'confidence_robustness_mean': np.mean(confidence_values)
        }


class CalibrationMetrics:
    """
    Calibration metrics including ECE, MCE, and reliability diagrams.
    Supports temperature scaling for calibration improvement.
    """
    
    def __init__(self, num_bins: int = 15):
        self.num_bins = num_bins
        self.reset()
        
    def reset(self):
        """Reset calibration accumulators."""
        self.all_confidences = []
        self.all_predictions = []
        self.all_targets = []
        
    def update(self, 
               outputs: torch.Tensor,
               targets: torch.Tensor):
        """
        Update calibration data with batch results.
        
        Args:
            outputs: Model logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        with torch.no_grad():
            probs = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            self.all_confidences.extend(confidences.cpu().numpy())
            self.all_predictions.extend(predictions.cpu().numpy())
            self.all_targets.extend(targets.cpu().numpy())
            
    def compute_ece_mce(self) -> Tuple[float, float]:
        """
        Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        
        Returns:
            Tuple of (ECE, MCE)
        """
        if not self.all_confidences:
            return 0.0, 0.0
            
        confidences = np.array(self.all_confidences)
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in current bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in bin
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
                
                # Average confidence in bin
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Calibration error for this bin
                bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                
                # Update ECE and MCE
                ece += bin_error * prop_in_bin
                mce = max(mce, bin_error)
                
        return ece, mce
    
    def get_reliability_diagram_data(self) -> Dict[str, np.ndarray]:
        """
        Get data for plotting reliability diagrams.
        
        Returns:
            Dict with bin_confidences, bin_accuracies, bin_counts
        """
        if not self.all_confidences:
            return {
                'bin_confidences': np.array([]),
                'bin_accuracies': np.array([]),
                'bin_counts': np.array([])
            }
            
        confidences = np.array(self.all_confidences)
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracy = (predictions[in_bin] == targets[in_bin]).mean()
                bin_confidence = confidences[in_bin].mean()
                bin_count = in_bin.sum()
                
                bin_confidences.append(bin_confidence)
                bin_accuracies.append(bin_accuracy)
                bin_counts.append(bin_count)
            else:
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(0.0)
                bin_counts.append(0)
                
        return {
            'bin_confidences': np.array(bin_confidences),
            'bin_accuracies': np.array(bin_accuracies),
            'bin_counts': np.array(bin_counts)
        }


class TemperatureScaling:
    """
    Temperature scaling for model calibration.
    Fits optimal temperature on validation set.
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False
        
    def fit(self, 
            logits: torch.Tensor,
            targets: torch.Tensor,
            lr: float = 0.01,
            max_iter: int = 50) -> float:
        """
        Fit temperature parameter on validation data.
        
        Args:
            logits: Model logits [num_samples, num_classes]
            targets: Ground truth labels [num_samples]
            lr: Learning rate for temperature optimization
            max_iter: Maximum optimization iterations
            
        Returns:
            Optimal temperature value
        """
        # Initialize temperature parameter
        temperature = torch.nn.Parameter(torch.ones(1) * 1.0)
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        
        def eval():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / temperature, targets)
            loss.backward()
            return loss
            
        optimizer.step(eval)
        
        self.temperature = temperature.item()
        self.is_fitted = True
        
        logger.info(f"Temperature scaling fitted: T = {self.temperature:.4f}")
        return self.temperature
        
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        if not self.is_fitted:
            logger.warning("Temperature scaling not fitted, returning original logits")
            return logits
            
        return logits / self.temperature


def compute_model_size(model: nn.Module) -> Tuple[int, float]:
    """
    Compute model parameter count and size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (param_count, size_mb)
    """
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate size (assuming float32 = 4 bytes)
    size_bytes = param_count * 4
    size_mb = size_bytes / (1024 * 1024)
    
    return param_count, size_mb


def get_system_info() -> Dict[str, Any]:
    """Get system information for reproducibility."""
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_version': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9
        })
        
    return info


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test core metrics
    core_metrics = CoreMetrics(num_classes=10)
    
    # Simulate some predictions
    batch_size = 32
    num_classes = 10
    
    for _ in range(5):
        # Random outputs and targets
        outputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        loss = F.cross_entropy(outputs, targets)
        
        core_metrics.update(outputs, targets, loss)
        
    results = core_metrics.compute()
    print("Core metrics:", results)
    
    # Test cost metrics
    cost_metrics = CostMetrics(warmup_batches=1)
    
    for _ in range(3):
        with cost_metrics.measure_batch():
            # Simulate computation
            time.sleep(0.01)
            dummy_tensor = torch.randn(1000, 1000)
            _ = dummy_tensor @ dummy_tensor.T
            
    cost_results = cost_metrics.compute()
    print("Cost metrics:", cost_results)
    
    # Test calibration metrics
    cal_metrics = CalibrationMetrics(num_bins=10)
    
    for _ in range(3):
        outputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        cal_metrics.update(outputs, targets)
        
    ece, mce = cal_metrics.compute_ece_mce()
    print(f"Calibration: ECE = {ece:.4f}, MCE = {mce:.4f}")
    
    # Test system info
    sys_info = get_system_info()
    print("System info:", sys_info)