"""
Regularization methods for continual learning.

This module implements Elastic Weight Consolidation (EWC) and Memory Aware Synapses (MAS)
regularization techniques to prevent catastrophic forgetting in continual learning scenarios.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import copy
from torch.utils.data import DataLoader
import numpy as np


@dataclass
class RegularizerConfig:
    """Configuration for regularization methods."""
    method: str = "ewc"  # "ewc" or "mas"
    lambda_reg: float = 1000.0  # Regularization strength
    fisher_estimation_samples: int = 1000  # Samples for Fisher estimation
    fisher_alpha: float = 0.9  # EMA factor for Fisher matrix updates
    mas_alpha: float = 0.9  # EMA factor for MAS importance updates
    regularize_gates: bool = True  # Whether to regularize gating network
    regularize_experts: bool = True  # Whether to regularize expert networks
    freeze_threshold: float = 0.8  # Threshold for freezing experts
    freeze_factor: float = 0.1  # Gradient scaling for frozen parameters


class EWCRegularizer:
    """
    Elastic Weight Consolidation (EWC) regularizer.
    
    Computes Fisher Information Matrix to estimate parameter importance
    and applies quadratic penalty to prevent forgetting.
    """
    
    def __init__(self, config: RegularizerConfig):
        self.config = config
        self.fisher_matrices: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.task_fishers: Dict[int, Dict[str, torch.Tensor]] = {}
        self.task_params: Dict[int, Dict[str, torch.Tensor]] = {}
        self._current_task = 0
        
    def estimate_fisher_matrix(self, 
                              model: nn.Module,
                              dataloader: DataLoader,
                              task_id: int,
                              device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Estimate Fisher Information Matrix for current task.
        
        Args:
            model: The neural network model
            dataloader: Data loader for current task
            task_id: Current task identifier
            device: Device to run computations on
            
        Returns:
            Dictionary mapping parameter names to Fisher diagonal estimates
        """
        model.eval()
        fisher_dict = {}
        
        # Initialize Fisher matrices
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param)
                
        samples_processed = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if samples_processed >= self.config.fisher_estimation_samples:
                break
                
            images, labels, _ = batch
            images, labels = images.to(device), labels.to(device)
            
            batch_size = images.size(0)
            
            # Forward pass
            outputs = model(images)
            
            # Sample from predicted distribution (for classification)
            if outputs.dim() == 2:  # Classification
                probs = torch.softmax(outputs, dim=1)
                sampled_labels = torch.multinomial(probs, 1).squeeze()
            else:
                sampled_labels = labels
                
            # Compute loss for sampled outputs
            loss = nn.CrossEntropyLoss()(outputs, sampled_labels)
            
            # Compute gradients
            model.zero_grad()
            loss.backward()
            
            # Accumulate Fisher information (squared gradients)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
                    
            samples_processed += batch_size
            
        # Normalize by number of samples
        for name in fisher_dict:
            fisher_dict[name] /= samples_processed
            
        return fisher_dict
        
    def update_fisher_matrix(self, 
                           model: nn.Module,
                           dataloader: DataLoader,
                           task_id: int,
                           device: torch.device):
        """Update Fisher matrix and optimal parameters for new task."""
        
        # Estimate Fisher matrix for current task
        current_fisher = self.estimate_fisher_matrix(model, dataloader, task_id, device)
        
        # Store task-specific Fisher and parameters
        self.task_fishers[task_id] = {}
        self.task_params[task_id] = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Store current parameters as optimal for this task
                self.task_params[task_id][name] = param.data.clone()
                self.task_fishers[task_id][name] = current_fisher[name].clone()
                
                # Update global Fisher matrix (EMA)
                if name in self.fisher_matrices:
                    self.fisher_matrices[name] = (
                        self.config.fisher_alpha * self.fisher_matrices[name] + 
                        (1 - self.config.fisher_alpha) * current_fisher[name]
                    )
                    self.optimal_params[name] = param.data.clone()
                else:
                    self.fisher_matrices[name] = current_fisher[name].clone()
                    self.optimal_params[name] = param.data.clone()
                    
        self._current_task = task_id
        
    def compute_penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty term.
        
        Args:
            model: The neural network model
            
        Returns:
            Regularization penalty scalar
        """
        penalty = 0.0
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher_matrices:
                # Skip if not regularizing this parameter type
                if not self._should_regularize_param(name):
                    continue
                    
                fisher = self.fisher_matrices[name]
                optimal = self.optimal_params[name]
                
                # Quadratic penalty weighted by Fisher information
                penalty += (fisher * (param - optimal) ** 2).sum()
                
        return self.config.lambda_reg * penalty
        
    def get_importance_weights(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Get parameter importance weights for visualization."""
        importance = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher_matrices:
                importance[name] = self.fisher_matrices[name].clone()
                
        return importance
        
    def _should_regularize_param(self, param_name: str) -> bool:
        """Check if parameter should be regularized based on config."""
        if "gate" in param_name.lower() or "routing" in param_name.lower():
            return self.config.regularize_gates
        elif "expert" in param_name.lower():
            return self.config.regularize_experts
        else:
            return True  # Regularize shared parameters by default


class MASRegularizer:
    """
    Memory Aware Synapses (MAS) regularizer.
    
    Estimates parameter importance based on gradient magnitude during training,
    independent of task-specific labels.
    """
    
    def __init__(self, config: RegularizerConfig):
        self.config = config
        self.importance_weights: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.accumulated_gradients: Dict[str, torch.Tensor] = {}
        self._gradient_steps = 0
        
    def accumulate_gradients(self, model: nn.Module, dataloader: DataLoader, device: torch.device):
        """
        Accumulate gradients for importance estimation.
        
        Args:
            model: The neural network model
            dataloader: Data loader for current task  
            device: Device to run computations on
        """
        model.train()
        
        # Initialize accumulation
        if not self.accumulated_gradients:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.accumulated_gradients[name] = torch.zeros_like(param)
                    
        samples_processed = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if samples_processed >= self.config.fisher_estimation_samples:
                break
                
            images, labels, _ = batch
            images = images.to(device)
            
            # Forward pass (no labels needed for MAS)
            outputs = model(images)
            
            # Compute L2 norm of output as unsupervised objective
            loss = torch.norm(outputs, dim=1).mean()
            
            # Compute gradients
            model.zero_grad()
            loss.backward()
            
            # Accumulate gradient magnitudes
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.accumulated_gradients[name] += torch.abs(param.grad.data)
                    
            samples_processed += images.size(0)
            self._gradient_steps += 1
            
    def update_importance_weights(self, model: nn.Module, task_id: int):
        """Update importance weights after task completion."""
        
        if not self.accumulated_gradients:
            return
            
        # Normalize by number of gradient steps
        for name in self.accumulated_gradients:
            self.accumulated_gradients[name] /= max(1, self._gradient_steps)
            
        # Update importance weights
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.accumulated_gradients:
                current_importance = self.accumulated_gradients[name]
                
                if name in self.importance_weights:
                    # EMA update
                    self.importance_weights[name] = (
                        self.config.mas_alpha * self.importance_weights[name] + 
                        (1 - self.config.mas_alpha) * current_importance
                    )
                else:
                    self.importance_weights[name] = current_importance.clone()
                    
                # Store optimal parameters
                self.optimal_params[name] = param.data.clone()
                
        # Reset accumulation
        self.accumulated_gradients.clear()
        self._gradient_steps = 0
        
    def compute_penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute MAS penalty term.
        
        Args:
            model: The neural network model
            
        Returns:
            Regularization penalty scalar
        """
        penalty = 0.0
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.importance_weights:
                # Skip if not regularizing this parameter type
                if not self._should_regularize_param(name):
                    continue
                    
                importance = self.importance_weights[name]
                optimal = self.optimal_params[name]
                
                # Quadratic penalty weighted by importance
                penalty += (importance * (param - optimal) ** 2).sum()
                
        return self.config.lambda_reg * penalty
        
    def get_importance_weights(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Get parameter importance weights for visualization."""
        return {name: weight.clone() for name, weight in self.importance_weights.items()}
        
    def _should_regularize_param(self, param_name: str) -> bool:
        """Check if parameter should be regularized based on config."""
        if "gate" in param_name.lower() or "routing" in param_name.lower():
            return self.config.regularize_gates
        elif "expert" in param_name.lower():
            return self.config.regularize_experts
        else:
            return True


class ParameterFreezer:
    """
    Manages selective parameter freezing based on importance scores.
    
    Freezes experts that are highly specialized for previous tasks.
    """
    
    def __init__(self, config: RegularizerConfig):
        self.config = config
        self.frozen_params: Dict[str, bool] = {}
        self.expert_activities: Dict[int, Dict[str, float]] = {}  # task_id -> expert_name -> activity
        
    def track_expert_activity(self, 
                             expert_activities: Dict[str, torch.Tensor],
                             task_id: int):
        """
        Track expert activation patterns for the current task.
        
        Args:
            expert_activities: Dictionary mapping expert names to activation tensors
            task_id: Current task identifier
        """
        if task_id not in self.expert_activities:
            self.expert_activities[task_id] = {}
            
        for expert_name, activations in expert_activities.items():
            # Compute mean activation for this expert
            mean_activation = float(activations.mean().item())
            
            if expert_name in self.expert_activities[task_id]:
                # Running average
                self.expert_activities[task_id][expert_name] = (
                    0.9 * self.expert_activities[task_id][expert_name] + 
                    0.1 * mean_activation
                )
            else:
                self.expert_activities[task_id][expert_name] = mean_activation
                
    def update_frozen_status(self, model: nn.Module, current_task: int):
        """
        Update frozen parameter status based on expert specialization.
        
        Args:
            model: The neural network model
            current_task: Current task ID
        """
        self.frozen_params.clear()
        
        if current_task <= 1 or not self.expert_activities:
            return  # No freezing for first task
            
        # Analyze expert specialization across tasks
        expert_specialization = self._compute_expert_specialization(current_task)
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Check if this parameter belongs to a specialized expert
            expert_name = self._get_expert_name_from_param(name)
            
            if expert_name and expert_name in expert_specialization:
                specialization = expert_specialization[expert_name]
                
                if specialization > self.config.freeze_threshold:
                    self.frozen_params[name] = True
                else:
                    self.frozen_params[name] = False
            else:
                self.frozen_params[name] = False
                
    def apply_gradient_scaling(self, model: nn.Module):
        """Apply gradient scaling to frozen parameters."""
        for name, param in model.named_parameters():
            if param.grad is not None and self.frozen_params.get(name, False):
                param.grad *= self.config.freeze_factor
                
    def _compute_expert_specialization(self, current_task: int) -> Dict[str, float]:
        """
        Compute specialization score for each expert.
        
        Higher scores indicate stronger specialization to previous tasks.
        """
        specialization = {}
        
        # Get all expert names
        all_experts = set()
        for task_activities in self.expert_activities.values():
            all_experts.update(task_activities.keys())
            
        for expert_name in all_experts:
            activities = []
            
            # Collect activities for previous tasks
            for task_id in range(1, current_task):
                if (task_id in self.expert_activities and 
                    expert_name in self.expert_activities[task_id]):
                    activities.append(self.expert_activities[task_id][expert_name])
                    
            if activities:
                # Specialization = max activity across previous tasks
                specialization[expert_name] = max(activities)
            else:
                specialization[expert_name] = 0.0
                
        return specialization
        
    def _get_expert_name_from_param(self, param_name: str) -> Optional[str]:
        """Extract expert identifier from parameter name."""
        parts = param_name.split('.')
        
        for part in parts:
            if 'expert' in part.lower():
                return part
                
        return None
        
    def get_frozen_parameters(self) -> List[str]:
        """Get list of currently frozen parameter names."""
        return [name for name, frozen in self.frozen_params.items() if frozen]


def create_regularizer(config: RegularizerConfig) -> Any:
    """Factory function to create appropriate regularizer."""
    if config.method.lower() == "ewc":
        return EWCRegularizer(config)
    elif config.method.lower() == "mas":
        return MASRegularizer(config)
    else:
        raise ValueError(f"Unknown regularization method: {config.method}")


class RegularizerManager:
    """
    Manages multiple regularization techniques and parameter freezing.
    
    Coordinates EWC/MAS regularization with expert freezing strategies.
    """
    
    def __init__(self, config: RegularizerConfig):
        self.config = config
        self.regularizer = create_regularizer(config)
        self.freezer = ParameterFreezer(config)
        self._current_task = 0
        
    def prepare_new_task(self, 
                        model: nn.Module,
                        dataloader: DataLoader, 
                        task_id: int,
                        device: torch.device):
        """Prepare regularizer for new task."""
        self._current_task = task_id
        
        if hasattr(self.regularizer, 'update_fisher_matrix'):
            # EWC regularizer
            if task_id > 1:  # Not first task
                self.regularizer.update_fisher_matrix(model, dataloader, task_id, device)
        elif hasattr(self.regularizer, 'accumulate_gradients'):
            # MAS regularizer - will accumulate during training
            pass
            
        # Update frozen parameter status
        self.freezer.update_frozen_status(model, task_id)
        
    def compute_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute total regularization loss."""
        if self._current_task <= 1:
            return torch.tensor(0.0, device=next(model.parameters()).device)
            
        return self.regularizer.compute_penalty(model)
        
    def post_training_update(self, 
                           model: nn.Module,
                           dataloader: Optional[DataLoader] = None,
                           device: Optional[torch.device] = None):
        """Update regularizer after task training completion."""
        if hasattr(self.regularizer, 'update_importance_weights'):
            # MAS regularizer
            if dataloader and device:
                self.regularizer.accumulate_gradients(model, dataloader, device)
            self.regularizer.update_importance_weights(model, self._current_task)
            
    def apply_gradient_constraints(self, model: nn.Module):
        """Apply gradient scaling for frozen parameters."""
        self.freezer.apply_gradient_scaling(model)
        
    def track_expert_activity(self, expert_activities: Dict[str, torch.Tensor]):
        """Track expert activities for freezing decisions."""
        self.freezer.track_expert_activity(expert_activities, self._current_task)
        
    def get_regularization_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get information about current regularization state."""
        info = {
            'method': self.config.method,
            'current_task': self._current_task,
            'lambda_reg': self.config.lambda_reg,
            'frozen_parameters': self.freezer.get_frozen_parameters(),
        }
        
        if hasattr(self.regularizer, 'get_importance_weights'):
            importance = self.regularizer.get_importance_weights(model)
            info['importance_stats'] = {
                name: {
                    'mean': float(weights.mean()),
                    'std': float(weights.std()),
                    'max': float(weights.max()),
                } for name, weights in importance.items()
            }
            
        return info