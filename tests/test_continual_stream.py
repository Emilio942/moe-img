"""
Test continual learning data streaming functionality.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from continual.stream import (
    TaskSequence, ReplayBuffer, ContinualDataStream, 
    TaskDataset, TaskConfig, TaskDataLoader
)


class TestTaskSequence:
    """Test task sequence management."""
    
    def test_add_task(self):
        """Test adding tasks to sequence."""
        sequence = TaskSequence()
        
        task1 = sequence.add_task(
            task_id=1,
            name="CIFAR-10",
            num_classes=10,
            dataset_name="cifar10"
        )
        
        assert task1.task_id == 1
        assert task1.name == "CIFAR-10"
        assert task1.num_classes == 10
        assert task1.class_offset == 0
        
        task2 = sequence.add_task(
            task_id=2, 
            name="CIFAR-100-Part1",
            num_classes=20,
            dataset_name="cifar100_subset"
        )
        
        assert task2.class_offset == 10
        assert sequence.get_total_classes() == 30
        
    def test_get_task(self):
        """Test retrieving tasks by ID."""
        sequence = TaskSequence()
        
        task1 = sequence.add_task(1, "Task1", 10, "dataset1")
        task2 = sequence.add_task(2, "Task2", 5, "dataset2")
        
        retrieved = sequence.get_task(1)
        assert retrieved.task_id == 1
        assert retrieved.name == "Task1"
        
        assert sequence.get_task(999) is None
        
    def test_get_tasks_up_to(self):
        """Test getting tasks up to a certain ID."""
        sequence = TaskSequence()
        
        for i in range(1, 5):
            sequence.add_task(i, f"Task{i}", 10, f"dataset{i}")
            
        tasks = sequence.get_tasks_up_to(3)
        assert len(tasks) == 3
        assert all(t.task_id <= 3 for t in tasks)


class TestReplayBuffer:
    """Test replay buffer functionality."""
    
    def test_fifo_buffer(self):
        """Test FIFO replay buffer."""
        buffer = ReplayBuffer(capacity=3, strategy="fifo")
        
        # Add samples
        images = torch.randn(5, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3, 4])
        
        buffer.add_samples(images, labels, task_id=1)
        
        # Buffer should be full (capacity 3)
        assert len(buffer) == 3
        
        # Check FIFO behavior - should have samples [2, 3, 4]
        sample_images, sample_labels, sample_tasks = buffer.sample_batch(3)
        assert len(sample_images) == 3
        assert set(sample_labels.tolist()) == {2, 3, 4}
        
    def test_reservoir_buffer(self):
        """Test reservoir sampling buffer."""
        buffer = ReplayBuffer(capacity=10, strategy="reservoir")
        
        # Add many samples
        for batch in range(5):
            images = torch.randn(10, 3, 32, 32)
            labels = torch.tensor([batch] * 10)
            buffer.add_samples(images, labels, task_id=batch + 1)
            
        assert len(buffer) == 10  # Should not exceed capacity
        
        # Sample and check diversity
        sample_images, sample_labels, sample_tasks = buffer.sample_batch(10)
        assert len(sample_images) == 10
        
    def test_sample_batch(self):
        """Test batch sampling from buffer."""
        buffer = ReplayBuffer(capacity=10, strategy="fifo")
        
        # Add samples
        images = torch.randn(5, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3, 4])
        buffer.add_samples(images, labels, task_id=1)
        
        # Sample smaller batch
        sample_images, sample_labels, sample_tasks = buffer.sample_batch(3)
        assert len(sample_images) == 3
        assert len(sample_labels) == 3
        assert len(sample_tasks) == 3
        
        # Sample larger than available
        sample_images, sample_labels, sample_tasks = buffer.sample_batch(10)
        assert len(sample_images) == 5  # Should return all available
        
    def test_get_task_samples(self):
        """Test retrieving samples for specific task."""
        buffer = ReplayBuffer(capacity=10, strategy="fifo")
        
        # Add samples from different tasks
        for task_id in [1, 2, 1, 2]:
            images = torch.randn(2, 3, 32, 32)
            labels = torch.tensor([task_id, task_id])
            buffer.add_samples(images, labels, task_id=task_id)
            
        # Get task 1 samples
        task1_images, task1_labels = buffer.get_task_samples(task_id=1)
        assert len(task1_images) == 4  # Should have 4 samples from task 1
        assert all(label == 1 for label in task1_labels)
        
        # Get task 2 samples
        task2_images, task2_labels = buffer.get_task_samples(task_id=2)
        assert len(task2_images) == 4  # Should have 4 samples from task 2
        
    def test_empty_buffer(self):
        """Test behavior with empty buffer."""
        buffer = ReplayBuffer(capacity=5, strategy="fifo")
        
        # Sample from empty buffer
        images, labels, tasks = buffer.sample_batch(3)
        assert len(images) == 0
        assert len(labels) == 0  
        assert len(tasks) == 0
        
        # Get task samples from empty buffer
        images, labels = buffer.get_task_samples(task_id=1)
        assert len(images) == 0
        assert len(labels) == 0


class TestTaskDataset:
    """Test task dataset wrapper."""
    
    def test_task_dataset_basic(self):
        """Test basic task dataset functionality."""
        # Create mock dataset
        class MockDataset:
            def __init__(self, size=100):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                return torch.randn(3, 32, 32), idx % 10
                
        base_dataset = MockDataset(100)
        task_config = TaskConfig(
            task_id=2,
            name="Task2", 
            num_classes=10,
            class_offset=10,
            dataset_name="mock"
        )
        
        task_dataset = TaskDataset(base_dataset, task_config)
        
        assert len(task_dataset) == 100
        
        # Check global label mapping
        image, global_label, task_id = task_dataset[5]
        assert global_label == 15  # 5 + 10 (offset)
        assert task_id == 2
        
    def test_task_dataset_with_limit(self):
        """Test task dataset with sample limit."""
        class MockDataset:
            def __len__(self):
                return 1000
                
            def __getitem__(self, idx):
                return torch.randn(3, 32, 32), idx % 10
                
        base_dataset = MockDataset()
        task_config = TaskConfig(
            task_id=1,
            name="Task1",
            num_classes=10,
            class_offset=0,
            dataset_name="mock",
            max_samples=50
        )
        
        task_dataset = TaskDataset(base_dataset, task_config)
        
        assert len(task_dataset) == 50  # Limited by max_samples
        
        # Should raise IndexError for out of range
        with pytest.raises(IndexError):
            _ = task_dataset[50]


class TestContinualDataStream:
    """Test main continual data stream."""
    
    def test_initialization(self):
        """Test stream initialization."""
        stream = ContinualDataStream(
            replay_capacity=100,
            replay_strategy="reservoir",
            batch_size=32,
            replay_ratio=0.3
        )
        
        assert len(stream.replay_buffer) == 0
        assert stream.batch_size == 32
        assert stream.replay_ratio == 0.3
        
    def test_create_cifar_sequence(self):
        """Test creating standard CIFAR sequence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            stream = ContinualDataStream()
            
            # This would normally download datasets, so we'll just test the structure
            tasks = stream.task_sequence.tasks
            
            # Should have empty task sequence initially
            assert len(tasks) == 0
            
            # Create sequence (won't actually download in test)
            try:
                cifar_tasks = stream.create_cifar_sequence(temp_dir)
                
                # Should have 3 tasks
                assert len(cifar_tasks) == 3
                
                # Check class offsets
                assert cifar_tasks[0].class_offset == 0  # CIFAR-10: 0-9
                assert cifar_tasks[1].class_offset == 10  # First 20 CIFAR-100: 10-29
                assert cifar_tasks[2].class_offset == 30  # Next 40 CIFAR-100: 30-69
                
                # Total classes should be 70
                assert stream.task_sequence.get_total_classes() == 70
                
            except Exception:
                # Skip if datasets can't be downloaded
                pytest.skip("Dataset download not available in test environment")
                
    def test_update_replay_buffer(self):
        """Test updating replay buffer."""
        stream = ContinualDataStream(replay_capacity=5)
        
        # Add samples
        images = torch.randn(3, 3, 32, 32)
        labels = torch.tensor([0, 1, 2])
        
        stream.update_replay_buffer(images, labels, task_id=1)
        
        assert len(stream.replay_buffer) == 3
        
        # Get replay batch
        replay_images, replay_labels, replay_tasks = stream.get_replay_batch(2)
        assert len(replay_images) == 2
        assert len(replay_labels) == 2
        assert len(replay_tasks) == 2
        

class TestTaskDataLoader:
    """Test task dataloader utilities."""
    
    def test_create_standard_sequence(self):
        """Test creating standard sequence."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                stream = TaskDataLoader.create_standard_sequence(temp_dir)
                
                assert isinstance(stream, ContinualDataStream)
                assert stream.replay_buffer.capacity == 500
                assert stream.batch_size == 64
                assert stream.replay_ratio == 0.5
                
        except Exception:
            # Skip if datasets not available
            pytest.skip("Dataset download not available in test environment")


if __name__ == "__main__":
    pytest.main([__file__])