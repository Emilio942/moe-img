"""
Continual learning trainer that integrates all components.

This module provides the main training interface for continual learning,
coordinating regularization, knowledge distillation, expert expansion, and metrics tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import copy
from pathlib import Path
import logging
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable
import numpy as np

from .stream import ContinualDataStream, TaskConfig
from .regularizers import RegularizerManager, RegularizerConfig
from .expansion import ExpertExpander, ExpansionConfig
from .distillation import KnowledgeDistiller, DistillationConfig
from .metrics import ContinualMetrics


@dataclass
class ContinualConfig:
    """Configuration for continual learning."""
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs_per_task: int = 10
    warmup_epochs: int = 2
    
    # Regularization
    regularizer_config: RegularizerConfig = None
    
    # Knowledge distillation
    distillation_config: DistillationConfig = None
    
    # Expert expansion
    expansion_config: ExpansionConfig = None
    
    # Replay and memory
    replay_ratio: float = 0.3
    use_replay: bool = True
    
    # Evaluation
    eval_every_epoch: bool = True
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints/continual"
    
    # Optimization
    optimizer_type: str = "adam"  # "adam", "sgd"
    weight_decay: float = 1e-4
    scheduler_type: str = "cosine"  # "cosine", "step", "none"
    
    # Logging
    log_level: str = "INFO"
    save_logs: bool = True
    log_dir: str = "./logs/continual"


class TaskCallback:
    """Base class for task-level callbacks."""
    
    def on_task_start(self, task_id: int, task_config: TaskConfig):
        """Called at the start of each task."""
        pass
        
    def on_task_end(self, task_id: int, metrics: Dict[str, Any]):
        """Called at the end of each task."""
        pass
        
    def on_epoch_start(self, task_id: int, epoch: int):
        """Called at the start of each epoch."""
        pass
        
    def on_epoch_end(self, task_id: int, epoch: int, metrics: Dict[str, Any]):
        """Called at the end of each epoch."""
        pass


class ContinualTrainer:
    """
    Main trainer for continual learning scenarios.
    
    Integrates regularization, distillation, expansion, and evaluation.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: ContinualConfig,
                 data_stream: ContinualDataStream,
                 device: torch.device = None):
        """
        Initialize continual trainer.
        
        Args:
            model: Base model to train
            config: Training configuration
            data_stream: Continual data stream manager
            device: Training device
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_stream = data_stream
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Initialize components
        self._init_components()
        
        # Training state
        self.current_task = 0
        self.training_history: List[Dict[str, Any]] = []
        self.callbacks: List[TaskCallback] = []
        
        # Setup logging
        self._setup_logging()
        
    def _init_components(self):
        """Initialize continual learning components."""
        # Regularizer
        if self.config.regularizer_config:
            self.regularizer = RegularizerManager(self.config.regularizer_config)
        else:
            self.regularizer = None
            
        # Knowledge distiller
        if self.config.distillation_config:
            self.distiller = KnowledgeDistiller(self.config.distillation_config)
        else:
            self.distiller = None
            
        # Expert expander
        if self.config.expansion_config:
            self.expander = ExpertExpander(self.config.expansion_config, self.model)
        else:
            self.expander = None
            
        # Metrics tracker
        self.metrics = ContinualMetrics()
        
        # Optimizer (will be reset for each task)
        self.optimizer = None
        self.scheduler = None
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'continual_training.log') if self.config.save_logs else logging.NullHandler(),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def add_callback(self, callback: TaskCallback):
        """Add a task callback."""
        self.callbacks.append(callback)
        
    def train_sequence(self, task_sequence: Optional[List[TaskConfig]] = None) -> Dict[str, Any]:
        """
        Train on a sequence of tasks.
        
        Args:
            task_sequence: List of task configurations. If None, uses default sequence.
            
        Returns:
            Training results dictionary
        """
        if task_sequence is None:
            # Use default CIFAR sequence
            task_sequence = self.data_stream.create_cifar_sequence()
            
        self.logger.info(f"Starting continual learning on {len(task_sequence)} tasks")
        
        results = {
            'task_results': {},
            'final_metrics': {},
            'training_history': []
        }
        
        for task_config in task_sequence:
            self.logger.info(f"Training on Task {task_config.task_id}: {task_config.name}")
            
            # Train on this task
            task_results = self.train_task(task_config)
            results['task_results'][task_config.task_id] = task_results
            
            # Evaluate on all previous tasks
            eval_results = self.evaluate_all_tasks(task_config.task_id)
            results['task_results'][task_config.task_id]['evaluations'] = eval_results
            
            # Update metrics
            for eval_task_id, accuracy in eval_results.items():
                self.metrics.update_task_performance(
                    evaluation_task=eval_task_id,
                    training_task=task_config.task_id,
                    accuracy=accuracy
                )
                
        # Compute final metrics
        results['final_metrics'] = self.metrics.compute_comprehensive_metrics(
            self.current_task
        )
        
        results['training_history'] = self.training_history
        
        self.logger.info("Continual learning sequence completed")
        return results
        
    def train_task(self, task_config: TaskConfig) -> Dict[str, Any]:
        """
        Train on a single task.
        
        Args:
            task_config: Configuration for the task
            
        Returns:
            Task training results
        """
        self.current_task = task_config.task_id
        
        # Notify callbacks
        for callback in self.callbacks:
            callback.on_task_start(task_config.task_id, task_config)
            
        # Setup data loaders
        train_loader = self.data_stream.get_task_dataloader(
            task_config.task_id, 
            train=True,
            include_replay=self.config.use_replay and task_config.task_id > 1
        )
        
        # Analyze task requirements and expand model if needed
        if self.expander and task_config.task_id > 1:
            analysis = self.expander.analyze_task_requirements(
                task_config.task_id,
                torch.cat([batch[0] for batch in train_loader][:5]),  # Sample for analysis
                self.model,
                self.device
            )
            
            self.model, new_experts = self.expander.expand_model(
                self.model, task_config.task_id, analysis
            )
            
            if new_experts:
                self.logger.info(f"Added {len(new_experts)} new experts: {new_experts}")
                
        # Setup optimizer and scheduler for this task
        self._setup_optimizer_for_task(task_config.task_id)
        
        # Prepare regularizer
        if self.regularizer:
            self.regularizer.prepare_new_task(
                self.model, train_loader, task_config.task_id, self.device
            )
            
        # Register teacher model for distillation (if not first task)
        if self.distiller and task_config.task_id > 1:
            teacher_model = copy.deepcopy(self.model)
            self.distiller.register_teacher_model(
                task_config.task_id - 1, teacher_model
            )
            
        # Training loop
        task_results = {
            'task_id': task_config.task_id,
            'epochs': [],
            'final_accuracy': 0.0,
            'best_accuracy': 0.0
        }
        
        best_accuracy = 0.0
        
        for epoch in range(self.config.epochs_per_task):
            # Notify callbacks
            for callback in self.callbacks:
                callback.on_epoch_start(task_config.task_id, epoch)
                
            # Train epoch
            epoch_metrics = self._train_epoch(train_loader, task_config.task_id, epoch)
            
            # Evaluate if requested
            if self.config.eval_every_epoch:
                eval_accuracy = self._evaluate_task(task_config.task_id)
                epoch_metrics['eval_accuracy'] = eval_accuracy
                
                if eval_accuracy > best_accuracy:
                    best_accuracy = eval_accuracy
                    
                    # Save best checkpoint
                    if self.config.save_checkpoints:
                        self._save_checkpoint(task_config.task_id, epoch, is_best=True)
                        
            task_results['epochs'].append(epoch_metrics)
            
            # Notify callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(task_config.task_id, epoch, epoch_metrics)
                
            self.logger.info(
                f"Task {task_config.task_id}, Epoch {epoch+1}/{self.config.epochs_per_task}: "
                f"Loss={epoch_metrics['loss']:.4f}, "
                f"Acc={epoch_metrics.get('eval_accuracy', 'N/A')}"
            )
            
        # Final evaluation
        final_accuracy = self._evaluate_task(task_config.task_id)
        task_results['final_accuracy'] = final_accuracy
        task_results['best_accuracy'] = best_accuracy
        
        # Post-training updates
        if self.regularizer:
            self.regularizer.post_training_update(self.model, train_loader, self.device)
            
        # Save final checkpoint
        if self.config.save_checkpoints:
            self._save_checkpoint(task_config.task_id, -1, is_final=True)
            
        # Record training history
        self.training_history.append(task_results)
        
        # Notify callbacks
        for callback in self.callbacks:
            callback.on_task_end(task_config.task_id, task_results)
            
        return task_results
        
    def _train_epoch(self, 
                    train_loader: DataLoader, 
                    task_id: int, 
                    epoch: int) -> Dict[str, Any]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Task {task_id} Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images, labels, task_ids = batch
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Add replay samples if enabled
            if self.config.use_replay and task_id > 1:
                replay_images, replay_labels, replay_task_ids = self.data_stream.get_replay_batch()
                
                if len(replay_images) > 0:
                    replay_images = replay_images.to(self.device)
                    replay_labels = replay_labels.to(self.device)
                    
                    # Mix current and replay samples
                    images = torch.cat([images, replay_images], dim=0)
                    labels = torch.cat([labels, replay_labels], dim=0)
                    
            # Forward pass
            outputs = self.model(images)
            
            # Task loss
            task_loss = nn.CrossEntropyLoss()(outputs, labels)
            
            # Regularization loss
            reg_loss = torch.tensor(0.0, device=self.device)
            if self.regularizer:
                reg_loss = self.regularizer.compute_regularization_loss(self.model)
                
            # Distillation loss
            dist_loss = torch.tensor(0.0, device=self.device)
            if self.distiller and task_id > 1:
                previous_tasks = list(range(1, task_id))
                dist_losses = self.distiller.compute_distillation_loss(
                    outputs, images, task_id, previous_tasks, self.device
                )
                dist_loss = dist_losses['total_distillation_loss']
                
            # Total loss
            total_batch_loss = (
                task_loss + 
                reg_loss +
                self.config.distillation_config.alpha * dist_loss 
                if self.config.distillation_config else task_loss + reg_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            
            # Apply gradient constraints
            if self.regularizer:
                self.regularizer.apply_gradient_constraints(self.model)
                
            self.optimizer.step()
            
            # Update replay buffer
            if self.config.use_replay:
                self.data_stream.update_replay_buffer(images[:len(batch[0])], labels[:len(batch[1])], task_id)
                
            total_loss += total_batch_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{total_batch_loss.item():.4f}",
                'Task Loss': f"{task_loss.item():.4f}",
                'Reg Loss': f"{reg_loss.item():.4f}",
                'Dist Loss': f"{dist_loss.item():.4f}"
            })
            
        # Update scheduler
        if self.scheduler:
            self.scheduler.step()
            
        return {
            'loss': total_loss / max(1, num_batches),
            'task_loss': task_loss.item(),
            'reg_loss': reg_loss.item(), 
            'dist_loss': dist_loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
    def _evaluate_task(self, task_id: int) -> float:
        """Evaluate on a specific task."""
        self.model.eval()
        
        test_loader = self.data_stream.get_task_dataloader(
            task_id, train=False, include_replay=False
        )
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                images, labels, _ = batch
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return accuracy
        
    def evaluate_all_tasks(self, up_to_task: int) -> Dict[int, float]:
        """Evaluate on all tasks up to the specified task."""
        results = {}
        
        for task_id in range(1, up_to_task + 1):
            accuracy = self._evaluate_task(task_id)
            results[task_id] = accuracy
            
        return results
        
    def _setup_optimizer_for_task(self, task_id: int):
        """Setup optimizer and scheduler for a task."""
        # Create optimizer
        if self.config.optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")
            
        # Create scheduler
        if self.config.scheduler_type.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs_per_task
            )
        elif self.config.scheduler_type.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.epochs_per_task // 3, gamma=0.1
            )
        else:
            self.scheduler = None
            
    def _save_checkpoint(self, 
                        task_id: int, 
                        epoch: int, 
                        is_best: bool = False,
                        is_final: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        suffix = ""
        if is_best:
            suffix = "_best"
        elif is_final:
            suffix = "_final"
            
        checkpoint_path = checkpoint_dir / f"continual_task{task_id}{suffix}.ckpt"
        
        checkpoint = {
            'task_id': task_id,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        self.training_history = checkpoint.get('training_history', [])
        self.current_task = checkpoint.get('task_id', 0)
        
        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
    def generate_continual_report(self, save_dir: str) -> Dict[str, str]:
        """Generate comprehensive continual learning report."""
        return self.metrics.generate_report(self.current_task, save_dir)
        
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.training_history:
            return {}
            
        summary = {
            'total_tasks': len(self.training_history),
            'current_task': self.current_task,
            'final_accuracies': [task['final_accuracy'] for task in self.training_history],
            'best_accuracies': [task['best_accuracy'] for task in self.training_history],
            'avg_final_accuracy': np.mean([task['final_accuracy'] for task in self.training_history]),
            'avg_best_accuracy': np.mean([task['best_accuracy'] for task in self.training_history])
        }
        
        # Add forgetting metrics if available
        if self.current_task > 1:
            forgetting_metrics = self.metrics.forgetting_tracker.compute_forgetting_metrics(
                self.current_task
            )
            summary['forgetting_metrics'] = forgetting_metrics
            
        return summary