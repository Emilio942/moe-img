"""
Metrics and tracking for continual learning evaluation.

This module provides comprehensive metrics for evaluating continual learning performance,
including forgetting measures, transfer analysis, and expert specialization tracking.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a specific time."""
    task_id: int
    epoch: int
    timestamp: float
    metrics: Dict[str, float]
    

class ForgettingTracker:
    """
    Tracks catastrophic forgetting across tasks.
    
    Measures how much performance degrades on previous tasks as new tasks are learned.
    """
    
    def __init__(self):
        self.task_accuracies: Dict[int, Dict[int, float]] = {}  # evaluation_task -> training_task -> accuracy
        self.peak_accuracies: Dict[int, float] = {}  # task_id -> peak_accuracy
        self.current_accuracies: Dict[int, float] = {}  # task_id -> current_accuracy
        self.forgetting_history: List[Dict[str, Any]] = []
        
    def update_accuracy(self, 
                       evaluation_task: int,
                       training_task: int, 
                       accuracy: float):
        """
        Update accuracy for a specific task evaluation.
        
        Args:
            evaluation_task: Task being evaluated on
            training_task: Task being trained on when evaluation happened
            accuracy: Accuracy achieved
        """
        if evaluation_task not in self.task_accuracies:
            self.task_accuracies[evaluation_task] = {}
            
        self.task_accuracies[evaluation_task][training_task] = accuracy
        
        # Update peak accuracy
        if evaluation_task not in self.peak_accuracies:
            self.peak_accuracies[evaluation_task] = accuracy
        else:
            self.peak_accuracies[evaluation_task] = max(
                self.peak_accuracies[evaluation_task], 
                accuracy
            )
            
        # Update current accuracy
        self.current_accuracies[evaluation_task] = accuracy
        
    def compute_forgetting_metrics(self, current_task: int) -> Dict[str, float]:
        """
        Compute forgetting metrics up to the current task.
        
        Returns:
            Dictionary containing various forgetting metrics
        """
        if current_task <= 1:
            return {'average_forgetting': 0.0, 'max_forgetting': 0.0}
            
        forgetting_values = []
        
        for task_id in range(1, current_task):
            if (task_id in self.peak_accuracies and 
                task_id in self.current_accuracies):
                
                peak_acc = self.peak_accuracies[task_id]
                current_acc = self.current_accuracies[task_id]
                forgetting = peak_acc - current_acc
                forgetting_values.append(forgetting)
                
        if not forgetting_values:
            return {'average_forgetting': 0.0, 'max_forgetting': 0.0}
            
        metrics = {
            'average_forgetting': np.mean(forgetting_values),
            'max_forgetting': np.max(forgetting_values),
            'std_forgetting': np.std(forgetting_values),
            'forgetting_per_task': {
                task_id: forgetting_values[task_id - 1] 
                for task_id in range(1, min(current_task, len(forgetting_values) + 1))
            }
        }
        
        # Record in history
        self.forgetting_history.append({
            'current_task': current_task,
            'metrics': metrics.copy()
        })
        
        return metrics
        
    def get_accuracy_matrix(self) -> np.ndarray:
        """
        Get accuracy matrix where entry (i,j) is accuracy on task i after training on task j.
        
        Returns:
            Accuracy matrix as numpy array
        """
        if not self.task_accuracies:
            return np.array([])
            
        max_task = max(max(self.task_accuracies.keys()), 
                      max(max(task_accs.keys()) for task_accs in self.task_accuracies.values()))
        
        matrix = np.full((max_task, max_task), np.nan)
        
        for eval_task, task_accs in self.task_accuracies.items():
            for train_task, accuracy in task_accs.items():
                matrix[eval_task - 1, train_task - 1] = accuracy
                
        return matrix
        
    def plot_forgetting_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot forgetting over time."""
        if not self.forgetting_history:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        tasks = [h['current_task'] for h in self.forgetting_history]
        avg_forgetting = [h['metrics']['average_forgetting'] for h in self.forgetting_history]
        max_forgetting = [h['metrics']['max_forgetting'] for h in self.forgetting_history]
        
        ax.plot(tasks, avg_forgetting, 'o-', label='Average Forgetting', linewidth=2)
        ax.plot(tasks, max_forgetting, 's-', label='Max Forgetting', linewidth=2)
        
        ax.set_xlabel('Task')
        ax.set_ylabel('Forgetting (Accuracy Drop)')
        ax.set_title('Catastrophic Forgetting Over Tasks')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class TransferAnalyzer:
    """
    Analyzes forward and backward transfer between tasks.
    
    Forward transfer: How much learning task A helps with learning task B.
    Backward transfer: How much learning task B affects performance on task A.
    """
    
    def __init__(self):
        self.baseline_performance: Dict[int, float] = {}  # task -> accuracy from scratch
        self.transfer_performance: Dict[int, Dict[int, float]] = {}  # target_task -> source_task -> accuracy
        self.learning_curves: Dict[Tuple[int, int], List[float]] = {}  # (source, target) -> accuracies over time
        
    def record_baseline_performance(self, task_id: int, accuracy: float):
        """Record baseline performance when learning from scratch."""
        self.baseline_performance[task_id] = accuracy
        
    def record_transfer_performance(self, 
                                  source_task: int,
                                  target_task: int, 
                                  accuracy: float):
        """Record performance when transferring from source to target task."""
        if target_task not in self.transfer_performance:
            self.transfer_performance[target_task] = {}
            
        self.transfer_performance[target_task][source_task] = accuracy
        
    def record_learning_curve(self, 
                             source_task: int,
                             target_task: int,
                             accuracies: List[float]):
        """Record learning curve for transfer scenario."""
        self.learning_curves[(source_task, target_task)] = accuracies.copy()
        
    def compute_forward_transfer(self, target_task: int) -> Dict[int, float]:
        """
        Compute forward transfer for target task.
        
        Returns:
            Dictionary mapping source tasks to transfer scores
        """
        if target_task not in self.baseline_performance:
            return {}
            
        baseline = self.baseline_performance[target_task]
        forward_transfer = {}
        
        if target_task in self.transfer_performance:
            for source_task, transfer_acc in self.transfer_performance[target_task].items():
                transfer_score = transfer_acc - baseline
                forward_transfer[source_task] = transfer_score
                
        return forward_transfer
        
    def compute_backward_transfer(self, source_task: int, up_to_task: int) -> List[float]:
        """
        Compute backward transfer for source task.
        
        Args:
            source_task: The source task to analyze
            up_to_task: Analyze up to this task
            
        Returns:
            List of backward transfer scores
        """
        if source_task not in self.baseline_performance:
            return []
            
        baseline = self.baseline_performance[source_task]
        backward_transfers = []
        
        for target_task in range(source_task + 1, up_to_task + 1):
            if (target_task in self.transfer_performance and 
                source_task in self.transfer_performance[target_task]):
                
                transfer_acc = self.transfer_performance[target_task][source_task]
                backward_transfer = transfer_acc - baseline
                backward_transfers.append(backward_transfer)
                
        return backward_transfers
        
    def compute_transfer_summary(self, up_to_task: int) -> Dict[str, Any]:
        """Compute comprehensive transfer analysis."""
        summary = {
            'forward_transfer': {},
            'backward_transfer': {},
            'average_forward_transfer': 0.0,
            'average_backward_transfer': 0.0
        }
        
        # Forward transfer analysis
        all_forward_transfers = []
        for task_id in range(2, up_to_task + 1):
            ft = self.compute_forward_transfer(task_id)
            summary['forward_transfer'][task_id] = ft
            all_forward_transfers.extend(ft.values())
            
        if all_forward_transfers:
            summary['average_forward_transfer'] = np.mean(all_forward_transfers)
            
        # Backward transfer analysis  
        all_backward_transfers = []
        for task_id in range(1, up_to_task):
            bt = self.compute_backward_transfer(task_id, up_to_task)
            summary['backward_transfer'][task_id] = bt
            all_backward_transfers.extend(bt)
            
        if all_backward_transfers:
            summary['average_backward_transfer'] = np.mean(all_backward_transfers)
            
        return summary
        
    def plot_transfer_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot transfer matrix as heatmap."""
        if not self.transfer_performance:
            return None
            
        # Build transfer matrix
        max_task = max(max(self.transfer_performance.keys()),
                      max(max(task_transfers.keys()) for task_transfers in self.transfer_performance.values()))
        
        transfer_matrix = np.full((max_task, max_task), np.nan)
        
        for target_task, source_transfers in self.transfer_performance.items():
            for source_task, accuracy in source_transfers.items():
                if target_task in self.baseline_performance:
                    baseline = self.baseline_performance[target_task]
                    transfer_score = accuracy - baseline
                    transfer_matrix[target_task - 1, source_task - 1] = transfer_score
                    
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(transfer_matrix, 
                   annot=True, 
                   fmt='.3f',
                   center=0,
                   cmap='RdBu_r',
                   ax=ax)
        
        ax.set_xlabel('Source Task')
        ax.set_ylabel('Target Task')
        ax.set_title('Transfer Learning Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class SpecializationTracker:
    """
    Tracks expert specialization patterns across tasks.
    
    Analyzes which experts become specialized for which tasks and how
    the specialization evolves over time.
    """
    
    def __init__(self):
        self.expert_activations: Dict[int, Dict[str, List[float]]] = {}  # task -> expert -> activations
        self.expert_task_mapping: Dict[str, Set[int]] = defaultdict(set)  # expert -> tasks
        self.specialization_history: List[Dict[str, Any]] = []
        self.gini_coefficients: Dict[int, Dict[str, float]] = {}  # task -> expert -> gini
        
    def update_expert_activations(self,
                                task_id: int,
                                expert_activations: Dict[str, torch.Tensor]):
        """
        Update expert activation statistics for a task.
        
        Args:
            task_id: Current task ID
            expert_activations: Dictionary mapping expert names to activation tensors
        """
        if task_id not in self.expert_activations:
            self.expert_activations[task_id] = {}
            
        for expert_name, activations in expert_activations.items():
            # Compute mean activation for this expert
            mean_activation = float(activations.mean().item())
            
            if expert_name not in self.expert_activations[task_id]:
                self.expert_activations[task_id][expert_name] = []
                
            self.expert_activations[task_id][expert_name].append(mean_activation)
            self.expert_task_mapping[expert_name].add(task_id)
            
    def compute_specialization_metrics(self, current_task: int) -> Dict[str, Any]:
        """
        Compute expert specialization metrics.
        
        Returns:
            Dictionary containing specialization analysis
        """
        metrics = {
            'expert_utilization': {},
            'task_coverage': {},
            'specialization_scores': {},
            'diversity_metrics': {}
        }
        
        # Get all experts
        all_experts = set()
        for task_experts in self.expert_activations.values():
            all_experts.update(task_experts.keys())
            
        # Expert utilization analysis
        for expert in all_experts:
            utilization_across_tasks = []
            
            for task_id in range(1, current_task + 1):
                if (task_id in self.expert_activations and 
                    expert in self.expert_activations[task_id]):
                    
                    avg_activation = np.mean(self.expert_activations[task_id][expert])
                    utilization_across_tasks.append(avg_activation)
                else:
                    utilization_across_tasks.append(0.0)
                    
            metrics['expert_utilization'][expert] = {
                'activations': utilization_across_tasks,
                'avg_activation': np.mean(utilization_across_tasks),
                'max_activation': np.max(utilization_across_tasks),
                'num_active_tasks': sum(1 for a in utilization_across_tasks if a > 0.1)
            }
            
            # Compute Gini coefficient for specialization
            gini = self._compute_gini_coefficient(utilization_across_tasks)
            metrics['specialization_scores'][expert] = gini
            
        # Task coverage analysis
        for task_id in range(1, current_task + 1):
            if task_id in self.expert_activations:
                active_experts = len([
                    expert for expert, activations in self.expert_activations[task_id].items()
                    if np.mean(activations) > 0.1
                ])
                
                metrics['task_coverage'][task_id] = {
                    'active_experts': active_experts,
                    'total_experts': len(all_experts),
                    'coverage_ratio': active_experts / max(1, len(all_experts))
                }
                
        # Diversity metrics
        metrics['diversity_metrics'] = {
            'avg_specialization': np.mean(list(metrics['specialization_scores'].values())),
            'specialization_std': np.std(list(metrics['specialization_scores'].values())),
            'highly_specialized_experts': sum(
                1 for gini in metrics['specialization_scores'].values() if gini > 0.7
            ),
            'generalist_experts': sum(
                1 for gini in metrics['specialization_scores'].values() if gini < 0.3
            )
        }
        
        # Record in history
        self.specialization_history.append({
            'current_task': current_task,
            'metrics': metrics.copy()
        })
        
        return metrics
        
    def _compute_gini_coefficient(self, values: List[float]) -> float:
        """
        Compute Gini coefficient for measuring inequality/specialization.
        
        Args:
            values: List of values (e.g., activations across tasks)
            
        Returns:
            Gini coefficient between 0 (perfect equality) and 1 (perfect inequality)
        """
        if not values or all(v == 0 for v in values):
            return 0.0
            
        values = np.array(values)
        values = np.abs(values)  # Ensure positive values
        
        if np.sum(values) == 0:
            return 0.0
            
        # Sort values
        sorted_values = np.sort(values)
        n = len(values)
        
        # Compute Gini coefficient
        cumsum = np.cumsum(sorted_values)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        return max(0.0, min(1.0, gini))
        
    def plot_specialization_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot expert specialization across tasks."""
        if not self.expert_activations:
            return None
            
        # Build activation matrix
        all_experts = sorted(set().union(*(
            task_experts.keys() for task_experts in self.expert_activations.values()
        )))
        all_tasks = sorted(self.expert_activations.keys())
        
        activation_matrix = np.zeros((len(all_experts), len(all_tasks)))
        
        for i, expert in enumerate(all_experts):
            for j, task_id in enumerate(all_tasks):
                if (task_id in self.expert_activations and 
                    expert in self.expert_activations[task_id]):
                    
                    activation_matrix[i, j] = np.mean(
                        self.expert_activations[task_id][expert]
                    )
                    
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(max(6, len(all_tasks)), max(4, len(all_experts) * 0.3)))
        
        sns.heatmap(activation_matrix,
                   xticklabels=[f'Task {t}' for t in all_tasks],
                   yticklabels=all_experts,
                   cmap='viridis',
                   ax=ax)
        
        ax.set_xlabel('Task')
        ax.set_ylabel('Expert')
        ax.set_title('Expert Specialization Across Tasks')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class ContinualMetrics:
    """
    Main metrics coordinator for continual learning evaluation.
    
    Combines forgetting tracking, transfer analysis, and specialization tracking.
    """
    
    def __init__(self):
        self.forgetting_tracker = ForgettingTracker()
        self.transfer_analyzer = TransferAnalyzer()
        self.specialization_tracker = SpecializationTracker()
        self.metric_history: List[MetricSnapshot] = []
        self.parameter_growth: List[Tuple[int, int]] = []  # (task_id, num_parameters)
        
    def update_task_performance(self,
                              evaluation_task: int,
                              training_task: int,
                              accuracy: float,
                              timestamp: float = None):
        """Update performance metrics for task evaluation."""
        self.forgetting_tracker.update_accuracy(evaluation_task, training_task, accuracy)
        
        if timestamp is None:
            timestamp = time.time()
            
        # Record metric snapshot
        snapshot = MetricSnapshot(
            task_id=training_task,
            epoch=-1,  # Task-level snapshot
            timestamp=timestamp,
            metrics={'accuracy': accuracy, 'evaluation_task': evaluation_task}
        )
        self.metric_history.append(snapshot)
        
    def update_expert_specialization(self,
                                   task_id: int, 
                                   expert_activations: Dict[str, torch.Tensor]):
        """Update expert specialization tracking."""
        self.specialization_tracker.update_expert_activations(task_id, expert_activations)
        
    def record_parameter_growth(self, task_id: int, num_parameters: int):
        """Record parameter growth over tasks."""
        self.parameter_growth.append((task_id, num_parameters))
        
    def compute_comprehensive_metrics(self, current_task: int) -> Dict[str, Any]:
        """
        Compute comprehensive continual learning metrics.
        
        Returns:
            Dictionary containing all metric categories
        """
        metrics = {
            'forgetting': self.forgetting_tracker.compute_forgetting_metrics(current_task),
            'transfer': self.transfer_analyzer.compute_transfer_summary(current_task),
            'specialization': self.specialization_tracker.compute_specialization_metrics(current_task),
            'parameter_growth': self._analyze_parameter_growth(),
            'task_id': current_task,
            'timestamp': time.time()
        }
        
        return metrics
        
    def _analyze_parameter_growth(self) -> Dict[str, Any]:
        """Analyze parameter growth over tasks."""
        if not self.parameter_growth:
            return {}
            
        tasks, param_counts = zip(*self.parameter_growth)
        
        return {
            'total_parameters': param_counts[-1] if param_counts else 0,
            'growth_over_tasks': list(param_counts),
            'growth_rate': (param_counts[-1] - param_counts[0]) / max(1, len(param_counts) - 1) 
                          if len(param_counts) > 1 else 0,
            'relative_growth': param_counts[-1] / param_counts[0] if param_counts[0] > 0 else 1
        }
        
    def generate_report(self, 
                       current_task: int,
                       save_dir: str) -> Dict[str, str]:
        """
        Generate comprehensive continual learning report.
        
        Returns:
            Dictionary mapping report types to file paths
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Compute metrics
        metrics = self.compute_comprehensive_metrics(current_task)
        
        # Generate plots
        plot_paths = {}
        
        # Forgetting curve
        fig = self.forgetting_tracker.plot_forgetting_curve()
        if fig:
            forgetting_path = save_path / 'forgetting_curve.png'
            plt.savefig(forgetting_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plot_paths['forgetting_curve'] = str(forgetting_path)
            
        # Transfer heatmap
        fig = self.transfer_analyzer.plot_transfer_heatmap()
        if fig:
            transfer_path = save_path / 'transfer_heatmap.png'
            plt.savefig(transfer_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plot_paths['transfer_heatmap'] = str(transfer_path)
            
        # Specialization heatmap
        fig = self.specialization_tracker.plot_specialization_heatmap()
        if fig:
            specialization_path = save_path / 'specialization_heatmap.png'
            plt.savefig(specialization_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plot_paths['specialization_heatmap'] = str(specialization_path)
            
        # Parameter growth plot
        if self.parameter_growth:
            fig, ax = plt.subplots(figsize=(8, 5))
            tasks, params = zip(*self.parameter_growth)
            ax.plot(tasks, params, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('Task')
            ax.set_ylabel('Number of Parameters')
            ax.set_title('Parameter Growth Over Tasks')
            ax.grid(True, alpha=0.3)
            
            growth_path = save_path / 'parameter_growth.png'
            plt.savefig(growth_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plot_paths['parameter_growth'] = str(growth_path)
            
        # Save metrics as JSON
        metrics_path = save_path / 'continual_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
            
        plot_paths['metrics_json'] = str(metrics_path)
        
        return plot_paths
        
    def export_data(self, save_path: str):
        """Export all collected data for external analysis."""
        data = {
            'forgetting_data': {
                'task_accuracies': self.forgetting_tracker.task_accuracies,
                'peak_accuracies': self.forgetting_tracker.peak_accuracies,
                'current_accuracies': self.forgetting_tracker.current_accuracies,
                'forgetting_history': self.forgetting_tracker.forgetting_history
            },
            'transfer_data': {
                'baseline_performance': self.transfer_analyzer.baseline_performance,
                'transfer_performance': self.transfer_analyzer.transfer_performance,
                'learning_curves': self.transfer_analyzer.learning_curves
            },
            'specialization_data': {
                'expert_activations': self.specialization_tracker.expert_activations,
                'expert_task_mapping': {k: list(v) for k, v in self.specialization_tracker.expert_task_mapping.items()},
                'specialization_history': self.specialization_tracker.specialization_history
            },
            'parameter_growth': self.parameter_growth,
            'metric_history': [
                {
                    'task_id': snapshot.task_id,
                    'epoch': snapshot.epoch,
                    'timestamp': snapshot.timestamp,
                    'metrics': snapshot.metrics
                } for snapshot in self.metric_history
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)