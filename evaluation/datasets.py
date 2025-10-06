"""
Dataset Management for Adaptive Evaluation Suite
=================================================

Handles in-domain, OOD, and corruption datasets for comprehensive evaluation.
Provides deterministic transformations and robust data loading.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CorruptionTransforms:
    """
    Deterministic corruption transformations for robustness evaluation.
    Implements CIFAR-10-C style corruptions with configurable severity.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.severity_levels = [1, 2, 3, 4, 5]
    
    def gaussian_noise(self, tensor: torch.Tensor, severity: int) -> torch.Tensor:
        """Add Gaussian noise with severity-dependent variance."""
        noise_variance = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
        noise = torch.randn_like(tensor) * noise_variance
        return torch.clamp(tensor + noise, 0, 1)
    
    def gaussian_blur(self, tensor: torch.Tensor, severity: int) -> torch.Tensor:
        """Apply Gaussian blur with severity-dependent kernel size."""
        kernel_size = [3, 5, 7, 9, 11][severity - 1]
        sigma = kernel_size / 6.0
        
        # Convert to PIL for consistent blur
        pil_img = TF.to_pil_image(tensor)
        blurred = TF.gaussian_blur(pil_img, kernel_size, sigma)
        return TF.to_tensor(blurred)
    
    def brightness_shift(self, tensor: torch.Tensor, severity: int) -> torch.Tensor:
        """Apply brightness shift with severity-dependent magnitude."""
        brightness_factor = [1.1, 1.2, 1.3, 1.4, 1.5][severity - 1]
        return torch.clamp(tensor * brightness_factor, 0, 1)
    
    def contrast_shift(self, tensor: torch.Tensor, severity: int) -> torch.Tensor:
        """Apply contrast adjustment with severity-dependent factor."""
        contrast_factor = [0.8, 0.7, 0.6, 0.5, 0.4][severity - 1]
        mean = tensor.mean(dim=[1, 2], keepdim=True)
        return torch.clamp((tensor - mean) * contrast_factor + mean, 0, 1)
    
    def random_crop_shift(self, tensor: torch.Tensor, severity: int) -> torch.Tensor:
        """Apply random crop with severity-dependent shift."""
        max_shift = [2, 4, 6, 8, 10][severity - 1]
        
        # Use deterministic random state
        shift_x = self.rng.randint(-max_shift, max_shift + 1)
        shift_y = self.rng.randint(-max_shift, max_shift + 1)
        
        # Apply shift by cropping and padding
        h, w = tensor.shape[-2:]
        
        # Calculate crop boundaries
        left = max(0, shift_x)
        top = max(0, shift_y) 
        right = min(w, w + shift_x)
        bottom = min(h, h + shift_y)
        
        # Crop and pad back to original size
        cropped = tensor[..., top:bottom, left:right]
        
        # Pad to restore original size
        pad_left = max(0, -shift_x)
        pad_right = max(0, shift_x - (w - right + left))
        pad_top = max(0, -shift_y)
        pad_bottom = max(0, shift_y - (h - bottom + top))
        
        padded = F.pad(cropped, (pad_left, pad_right, pad_top, pad_bottom), 
                      mode='reflect')
        
        return padded
    
    def get_corruption(self, corruption_type: str) -> Callable:
        """Get corruption function by name."""
        corruptions = {
            'gaussian_noise': self.gaussian_noise,
            'gaussian_blur': self.gaussian_blur,
            'brightness': self.brightness_shift,
            'contrast': self.contrast_shift,
            'crop_shift': self.random_crop_shift,
        }
        
        if corruption_type not in corruptions:
            raise ValueError(f"Unknown corruption: {corruption_type}")
        
        return corruptions[corruption_type]


class CorruptedDataset(Dataset):
    """
    Dataset wrapper that applies corruption transformations.
    Maintains deterministic behavior with seeded transformations.
    """
    
    def __init__(self, 
                 base_dataset: Dataset,
                 corruption_type: str,
                 severity: int,
                 seed: int = 42):
        self.base_dataset = base_dataset
        self.corruption_type = corruption_type
        self.severity = severity
        self.transforms = CorruptionTransforms(seed)
        self.corruption_fn = self.transforms.get_corruption(corruption_type)
        
        # Create deterministic hash for dataset identity
        self.dataset_hash = self._compute_dataset_hash()
        
    def _compute_dataset_hash(self) -> str:
        """Compute deterministic hash for dataset configuration."""
        hash_input = f"{self.corruption_type}_{self.severity}_{len(self.base_dataset)}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.base_dataset[idx]
        
        # Ensure image is tensor
        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)
            
        # Apply corruption
        corrupted_image = self.corruption_fn(image, self.severity)
        
        return corrupted_image, label


class EvaluationDatasets:
    """
    Comprehensive dataset manager for adaptive evaluation.
    Handles in-domain, OOD, and corruption datasets with reproducible splits.
    """
    
    def __init__(self, 
                 data_root: str = "./data",
                 batch_size: int = 256,
                 num_workers: int = 4,
                 seed: int = 42):
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        # Standard CIFAR-10 transforms
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        # Corruption types for OOD evaluation
        self.corruption_types = [
            'gaussian_noise',
            'gaussian_blur', 
            'brightness',
            'contrast',
            'crop_shift'
        ]
        
        self.severity_levels = [1, 2, 3, 4, 5]
        
        logger.info(f"EvaluationDatasets initialized with seed {seed}")
        
    def get_in_domain_dataset(self) -> Tuple[DataLoader, Dict[str, Any]]:
        """
        Load in-domain test dataset (CIFAR-10).
        
        Returns:
            DataLoader and metadata dict
        """
        test_dataset = datasets.CIFAR10(
            root=self.data_root,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        metadata = {
            'dataset': 'CIFAR-10',
            'split': 'test',
            'num_samples': len(test_dataset),
            'num_classes': 10,
            'batch_size': self.batch_size,
            'hash': self._compute_loader_hash(test_dataset)
        }
        
        logger.info(f"In-domain dataset loaded: {metadata}")
        return test_loader, metadata
        
    def get_ood_datasets(self) -> Dict[str, Tuple[DataLoader, Dict[str, Any]]]:
        """
        Create OOD corruption datasets for robustness evaluation.
        
        Returns:
            Dict mapping corruption names to (DataLoader, metadata)
        """
        # Base test dataset without normalization for corruptions
        base_transform_no_norm = transforms.ToTensor()
        base_test_dataset = datasets.CIFAR10(
            root=self.data_root,
            train=False,
            download=True,
            transform=base_transform_no_norm
        )
        
        ood_datasets = {}
        
        for corruption_type in self.corruption_types:
            for severity in self.severity_levels:
                # Create corrupted dataset
                corrupted_dataset = CorruptedDataset(
                    base_test_dataset,
                    corruption_type,
                    severity,
                    seed=self.seed
                )
                
                # Apply normalization after corruption
                normalized_dataset = NormalizedDataset(
                    corrupted_dataset,
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010)
                )
                
                # Create DataLoader
                loader = DataLoader(
                    normalized_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True
                )
                
                # Metadata
                corruption_name = f"{corruption_type}_severity_{severity}"
                metadata = {
                    'dataset': 'CIFAR-10-C',
                    'corruption_type': corruption_type,
                    'severity': severity,
                    'num_samples': len(corrupted_dataset),
                    'batch_size': self.batch_size,
                    'hash': corrupted_dataset.dataset_hash
                }
                
                ood_datasets[corruption_name] = (loader, metadata)
                
        logger.info(f"Created {len(ood_datasets)} OOD corruption datasets")
        return ood_datasets
    
    def get_validation_split(self, 
                           split_ratio: float = 0.1) -> Tuple[DataLoader, Dict[str, Any]]:
        """
        Create validation split for temperature scaling and calibration.
        
        Args:
            split_ratio: Fraction of training data to use for validation
            
        Returns:
            Validation DataLoader and metadata
        """
        # Load training dataset
        train_dataset = datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=True,
            transform=self.test_transform
        )
        
        # Deterministic split
        generator = torch.Generator().manual_seed(self.seed)
        val_size = int(len(train_dataset) * split_ratio)
        train_size = len(train_dataset) - val_size
        
        _, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size], generator=generator
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        metadata = {
            'dataset': 'CIFAR-10',
            'split': 'validation',
            'split_ratio': split_ratio,
            'num_samples': len(val_dataset),
            'batch_size': self.batch_size,
            'seed': self.seed
        }
        
        logger.info(f"Validation split created: {metadata}")
        return val_loader, metadata
        
    def _compute_loader_hash(self, dataset: Dataset) -> str:
        """Compute hash for dataset reproducibility verification."""
        hash_input = f"{type(dataset).__name__}_{len(dataset)}_{self.seed}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]


class NormalizedDataset(Dataset):
    """Dataset wrapper that applies normalization after corruption."""
    
    def __init__(self, 
                 base_dataset: Dataset,
                 mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]):
        self.base_dataset = base_dataset
        self.normalize = transforms.Normalize(mean, std)
        
    def __len__(self) -> int:
        return len(self.base_dataset)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.base_dataset[idx]
        normalized_image = self.normalize(image)
        return normalized_image, label


def create_ood_splits(datasets_manager: EvaluationDatasets,
                     max_corruptions: Optional[int] = None) -> Dict[str, Any]:
    """
    Create comprehensive OOD splits for evaluation.
    
    Args:
        datasets_manager: EvaluationDatasets instance
        max_corruptions: Limit number of corruptions for quick evaluation
        
    Returns:
        Dict with in-domain and OOD datasets
    """
    logger.info("Creating comprehensive evaluation splits...")
    
    # Get in-domain dataset
    in_domain_loader, in_domain_meta = datasets_manager.get_in_domain_dataset()
    
    # Get OOD datasets
    ood_datasets = datasets_manager.get_ood_datasets()
    
    # Optionally subsample for quick evaluation
    if max_corruptions is not None:
        logger.info(f"Limiting to {max_corruptions} corruption configurations")
        ood_items = list(ood_datasets.items())[:max_corruptions]
        ood_datasets = dict(ood_items)
    
    # Get validation split for calibration
    val_loader, val_meta = datasets_manager.get_validation_split()
    
    evaluation_splits = {
        'in_domain': {
            'loader': in_domain_loader,
            'metadata': in_domain_meta
        },
        'validation': {
            'loader': val_loader,
            'metadata': val_meta
        },
        'ood_datasets': ood_datasets,
        'summary': {
            'total_ood_splits': len(ood_datasets),
            'corruption_types': datasets_manager.corruption_types,
            'severity_levels': datasets_manager.severity_levels,
            'seed': datasets_manager.seed
        }
    }
    
    logger.info(f"Created evaluation splits: "
               f"1 in-domain, 1 validation, {len(ood_datasets)} OOD")
    
    return evaluation_splits


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test dataset creation
    datasets_manager = EvaluationDatasets(seed=42)
    
    # Test in-domain dataset
    in_domain_loader, metadata = datasets_manager.get_in_domain_dataset()
    print(f"In-domain dataset: {metadata}")
    
    # Test one batch
    for batch_idx, (images, labels) in enumerate(in_domain_loader):
        print(f"Batch {batch_idx}: images {images.shape}, labels {labels.shape}")
        if batch_idx >= 2:  # Test first few batches
            break
    
    # Test OOD datasets (limited)
    ood_datasets = datasets_manager.get_ood_datasets()
    print(f"Created {len(ood_datasets)} OOD datasets")
    
    # Test one OOD dataset
    first_ood_name = list(ood_datasets.keys())[0]
    first_ood_loader, first_ood_meta = ood_datasets[first_ood_name]
    print(f"First OOD dataset ({first_ood_name}): {first_ood_meta}")
    
    # Test comprehensive splits
    splits = create_ood_splits(datasets_manager, max_corruptions=5)
    print(f"Evaluation splits summary: {splits['summary']}")