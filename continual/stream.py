"""
Data streaming and task management for continual learning.

This module provides the infrastructure for managing sequential tasks and data streams
in continual learning scenarios. It includes task sequencing, replay buffers, and
data loading utilities for incremental learning.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Iterator, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import random
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision
import torchvision.transforms as transforms


@dataclass
class TaskConfig:
    """Configuration for a single task."""
    task_id: int
    name: str
    num_classes: int
    class_offset: int  # Global label offset
    dataset_name: str
    data_path: Optional[str] = None
    transform: Optional[Any] = None
    max_samples: Optional[int] = None
    
    
class TaskSequence:
    """
    Manages a sequence of tasks for continual learning.
    
    Provides consistent global label space expansion and task metadata.
    """
    
    def __init__(self):
        self.tasks: List[TaskConfig] = []
        self._current_offset = 0
        
    def add_task(self, 
                 task_id: int,
                 name: str, 
                 num_classes: int,
                 dataset_name: str,
                 data_path: Optional[str] = None,
                 transform: Optional[Any] = None,
                 max_samples: Optional[int] = None) -> TaskConfig:
        """Add a new task to the sequence."""
        task = TaskConfig(
            task_id=task_id,
            name=name,
            num_classes=num_classes,
            class_offset=self._current_offset,
            dataset_name=dataset_name,
            data_path=data_path,
            transform=transform,
            max_samples=max_samples
        )
        
        self.tasks.append(task)
        self._current_offset += num_classes
        return task
        
    def get_task(self, task_id: int) -> Optional[TaskConfig]:
        """Get task configuration by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
        
    def get_total_classes(self) -> int:
        """Get total number of classes across all tasks."""
        return self._current_offset
        
    def get_tasks_up_to(self, task_id: int) -> List[TaskConfig]:
        """Get all tasks up to and including the given task ID."""
        return [task for task in self.tasks if task.task_id <= task_id]


class ReplayBuffer:
    """
    Memory buffer for storing examples from previous tasks.
    
    Supports FIFO and reservoir sampling strategies.
    """
    
    def __init__(self, 
                 capacity: int,
                 strategy: str = "reservoir"):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of samples to store
            strategy: Sampling strategy ("fifo" or "reservoir")
        """
        self.capacity = capacity
        self.strategy = strategy
        self.buffer: List[Tuple[torch.Tensor, int, int]] = []  # (image, label, task_id)
        self._insertion_count = 0
        
    def add_samples(self, 
                   images: torch.Tensor,
                   labels: torch.Tensor, 
                   task_id: int):
        """Add samples to the buffer."""
        batch_size = images.size(0)
        
        for i in range(batch_size):
            sample = (images[i].clone(), labels[i].item(), task_id)
            
            if len(self.buffer) < self.capacity:
                # Buffer not full yet
                self.buffer.append(sample)
            else:
                if self.strategy == "fifo":
                    # Replace oldest sample
                    idx = self._insertion_count % self.capacity
                    self.buffer[idx] = sample
                elif self.strategy == "reservoir":
                    # Reservoir sampling
                    j = random.randint(0, self._insertion_count)
                    if j < self.capacity:
                        self.buffer[j] = sample
                        
            self._insertion_count += 1
            
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch from the buffer."""
        if len(self.buffer) == 0:
            return torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
            
        indices = random.sample(range(len(self.buffer)), 
                               min(batch_size, len(self.buffer)))
        
        images = torch.stack([self.buffer[i][0] for i in indices])
        labels = torch.tensor([self.buffer[i][1] for i in indices], dtype=torch.long)
        task_ids = torch.tensor([self.buffer[i][2] for i in indices], dtype=torch.long)
        
        return images, labels, task_ids
        
    def get_task_samples(self, task_id: int, max_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all samples for a specific task."""
        task_samples = [(img, label) for img, label, tid in self.buffer if tid == task_id]
        
        if max_samples and len(task_samples) > max_samples:
            task_samples = random.sample(task_samples, max_samples)
            
        if not task_samples:
            return torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)
            
        images = torch.stack([sample[0] for sample in task_samples])
        labels = torch.tensor([sample[1] for sample in task_samples], dtype=torch.long)
        
        return images, labels
        
    def __len__(self) -> int:
        return len(self.buffer)
        
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self._insertion_count = 0


class TaskDataset(Dataset):
    """Dataset wrapper that adds task information."""
    
    def __init__(self, 
                 dataset: Dataset,
                 task_config: TaskConfig):
        self.dataset = dataset
        self.task_config = task_config
        
    def __len__(self):
        if self.task_config.max_samples:
            return min(len(self.dataset), self.task_config.max_samples)
        return len(self.dataset)
        
    def __getitem__(self, idx):
        if self.task_config.max_samples and idx >= self.task_config.max_samples:
            raise IndexError("Index out of range")
            
        image, label = self.dataset[idx]
        
        # Apply task-specific transform if provided
        if self.task_config.transform:
            image = self.task_config.transform(image)
            
        # Adjust label to global label space
        global_label = label + self.task_config.class_offset
        
        return image, global_label, self.task_config.task_id


class ContinualDataStream:
    """
    Main interface for continual learning data streaming.
    
    Manages task sequences, replay buffers, and data loading.
    """
    
    def __init__(self, 
                 replay_capacity: int = 500,
                 replay_strategy: str = "reservoir",
                 batch_size: int = 64,
                 replay_ratio: float = 0.5):
        """
        Initialize continual data stream.
        
        Args:
            replay_capacity: Maximum samples in replay buffer
            replay_strategy: Replay sampling strategy
            batch_size: Batch size for training
            replay_ratio: Ratio of replay samples in mixed batches (0-1)
        """
        self.task_sequence = TaskSequence()
        self.replay_buffer = ReplayBuffer(replay_capacity, replay_strategy)
        self.batch_size = batch_size
        self.replay_ratio = replay_ratio
        self._current_task_id = 0
        
    def create_cifar_sequence(self, data_path: str = "./data") -> List[TaskConfig]:
        """
        Create a standard CIFAR-10 -> CIFAR-100 task sequence.
        
        Task 1: CIFAR-10 (10 classes)
        Task 2: 20 new classes from CIFAR-100
        Task 3: 40 more classes from CIFAR-100
        """
        # Standard transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Task 1: CIFAR-10
        task1 = self.task_sequence.add_task(
            task_id=1,
            name="CIFAR-10",
            num_classes=10,
            dataset_name="cifar10",
            data_path=data_path,
            transform=transform_train
        )
        
        # Task 2: First 20 classes of CIFAR-100
        task2 = self.task_sequence.add_task(
            task_id=2, 
            name="CIFAR-100-Part1",
            num_classes=20,
            dataset_name="cifar100_subset_0_20",
            data_path=data_path,
            transform=transform_train
        )
        
        # Task 3: Next 40 classes of CIFAR-100  
        task3 = self.task_sequence.add_task(
            task_id=3,
            name="CIFAR-100-Part2", 
            num_classes=40,
            dataset_name="cifar100_subset_20_60",
            data_path=data_path,
            transform=transform_train
        )
        
        return [task1, task2, task3]
        
    def get_task_dataloader(self, 
                           task_id: int,
                           train: bool = True,
                           include_replay: bool = False) -> DataLoader:
        """
        Get data loader for a specific task.
        
        Args:
            task_id: Task identifier
            train: Whether to get training or test data
            include_replay: Whether to mix in replay samples
        """
        task_config = self.task_sequence.get_task(task_id)
        if not task_config:
            raise ValueError(f"Task {task_id} not found")
            
        # Load base dataset
        dataset = self._load_dataset(task_config, train=train)
        task_dataset = TaskDataset(dataset, task_config)
        
        if include_replay and train and len(self.replay_buffer) > 0:
            # Create mixed dataset with replay samples
            dataloader = self._create_mixed_dataloader(task_dataset)
        else:
            dataloader = DataLoader(
                task_dataset,
                batch_size=self.batch_size,
                shuffle=train,
                num_workers=2,
                pin_memory=True
            )
            
        return dataloader
        
    def _load_dataset(self, task_config: TaskConfig, train: bool = True) -> Dataset:
        """Load dataset based on task configuration."""
        data_path = task_config.data_path or "./data"
        
        if task_config.dataset_name == "cifar10":
            return torchvision.datasets.CIFAR10(
                root=data_path,
                train=train,
                download=True,
                transform=None  # Transform applied in TaskDataset
            )
        elif task_config.dataset_name.startswith("cifar100_subset"):
            # Parse subset range from name
            parts = task_config.dataset_name.split("_")
            start_class = int(parts[2])
            end_class = int(parts[3])
            
            full_dataset = torchvision.datasets.CIFAR100(
                root=data_path,
                train=train,
                download=True,
                transform=None
            )
            
            # Filter classes
            indices = [i for i, (_, label) in enumerate(full_dataset) 
                      if start_class <= label < end_class]
            
            # Create subset with remapped labels
            class SubsetDataset(Dataset):
                def __init__(self, dataset, indices, start_class):
                    self.dataset = dataset
                    self.indices = indices
                    self.start_class = start_class
                    
                def __len__(self):
                    return len(self.indices)
                    
                def __getitem__(self, idx):
                    original_idx = self.indices[idx]
                    image, label = self.dataset[original_idx]
                    # Remap label to start from 0
                    new_label = label - self.start_class
                    return image, new_label
                    
            return SubsetDataset(full_dataset, indices, start_class)
        else:
            raise ValueError(f"Unknown dataset: {task_config.dataset_name}")
            
    def _create_mixed_dataloader(self, task_dataset: TaskDataset) -> DataLoader:
        """Create dataloader that mixes current task and replay samples."""
        
        class MixedSampler:
            def __init__(self, task_dataset, replay_buffer, batch_size, replay_ratio):
                self.task_dataset = task_dataset
                self.replay_buffer = replay_buffer
                self.batch_size = batch_size
                self.replay_ratio = replay_ratio
                
                # Calculate splits
                self.replay_samples = int(batch_size * replay_ratio)
                self.task_samples = batch_size - self.replay_samples
                
            def __iter__(self):
                task_indices = list(range(len(self.task_dataset)))
                random.shuffle(task_indices)
                
                for i in range(0, len(task_indices), self.task_samples):
                    batch_indices = task_indices[i:i + self.task_samples]
                    yield batch_indices
                    
            def __len__(self):
                return (len(self.task_dataset) + self.task_samples - 1) // self.task_samples
        
        # For now, return simple dataloader - mixed sampling will be handled in trainer
        return DataLoader(
            task_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
    def update_replay_buffer(self, 
                           images: torch.Tensor,
                           labels: torch.Tensor,
                           task_id: int):
        """Add samples to replay buffer."""
        self.replay_buffer.add_samples(images, labels, task_id)
        
    def get_replay_batch(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a batch of replay samples."""
        if batch_size is None:
            batch_size = max(1, int(self.batch_size * self.replay_ratio))
        return self.replay_buffer.sample_batch(batch_size)
        
    def get_evaluation_dataloaders(self, up_to_task: int) -> Dict[int, DataLoader]:
        """Get evaluation dataloaders for all tasks up to the specified task."""
        dataloaders = {}
        for task_config in self.task_sequence.get_tasks_up_to(up_to_task):
            dataloaders[task_config.task_id] = self.get_task_dataloader(
                task_config.task_id, 
                train=False, 
                include_replay=False
            )
        return dataloaders


class TaskDataLoader:
    """Utility class for managing task-specific data loading."""
    
    @staticmethod
    def create_standard_sequence(data_path: str = "./data") -> ContinualDataStream:
        """Create a standard continual learning sequence."""
        stream = ContinualDataStream(
            replay_capacity=500,
            replay_strategy="reservoir", 
            batch_size=64,
            replay_ratio=0.5
        )
        
        stream.create_cifar_sequence(data_path)
        return stream
        
    @staticmethod
    def get_task_stats(dataloader: DataLoader) -> Dict[str, Any]:
        """Get statistics about a task dataloader."""
        total_samples = 0
        class_counts = defaultdict(int)
        task_ids = set()
        
        for batch in dataloader:
            images, labels, tids = batch
            total_samples += len(images)
            
            for label in labels:
                class_counts[label.item()] += 1
                
            for tid in tids:
                task_ids.add(tid.item())
                
        return {
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'class_distribution': dict(class_counts),
            'task_ids': list(task_ids)
        }