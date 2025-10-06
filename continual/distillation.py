"""
Knowledge distillation for continual learning.

This module implements knowledge distillation techniques to transfer knowledge
from previous task models to new task models, helping prevent catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import copy
from collections import defaultdict
import numpy as np


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    temperature: float = 4.0  # Temperature for softmax distillation
    alpha: float = 0.5  # Weight for distillation loss vs task loss
    beta: float = 0.3  # Weight for feature distillation vs output distillation
    use_feature_distillation: bool = True  # Whether to use intermediate feature distillation
    use_attention_distillation: bool = True  # Whether to distill attention patterns
    replay_distillation: bool = True  # Use replay buffer for distillation
    distill_gate: bool = True  # Distill gating network outputs
    adaptive_temperature: bool = False  # Adaptive temperature based on task similarity
    min_temperature: float = 2.0  # Minimum temperature for adaptive mode
    max_temperature: float = 8.0  # Maximum temperature for adaptive mode


class SoftTargetGenerator:
    """
    Generates soft targets from previous task models for knowledge distillation.
    """
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.teacher_models: Dict[int, nn.Module] = {}  # task_id -> model
        self.class_mappings: Dict[int, Dict[int, int]] = {}  # task_id -> old_class -> new_class
        self.task_temperatures: Dict[int, float] = {}  # Adaptive temperatures per task
        
    def register_teacher_model(self, 
                              task_id: int, 
                              model: nn.Module,
                              class_mapping: Optional[Dict[int, int]] = None):
        """
        Register a teacher model for a completed task.
        
        Args:
            task_id: Task identifier
            model: Trained model for this task
            class_mapping: Mapping from task-specific to global class indices
        """
        # Store a copy of the model in eval mode
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        
        # Freeze all parameters
        for param in teacher_model.parameters():
            param.requires_grad = False
            
        self.teacher_models[task_id] = teacher_model
        
        if class_mapping:
            self.class_mappings[task_id] = class_mapping
            
        # Initialize temperature
        self.task_temperatures[task_id] = self.config.temperature
        
    def generate_soft_targets(self,
                            inputs: torch.Tensor,
                            target_tasks: List[int],
                            device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Generate soft targets from teacher models.
        
        Args:
            inputs: Input batch
            target_tasks: List of task IDs to generate targets for
            device: Device for computation
            
        Returns:
            Dictionary containing various soft targets
        """
        soft_targets = {}
        
        inputs = inputs.to(device)
        
        for task_id in target_tasks:
            if task_id not in self.teacher_models:
                continue
                
            teacher = self.teacher_models[task_id].to(device)
            temperature = self.task_temperatures[task_id]
            
            with torch.no_grad():
                # Generate output targets
                teacher_outputs = teacher(inputs)
                
                if isinstance(teacher_outputs, tuple):
                    # Handle models that return multiple outputs
                    teacher_logits = teacher_outputs[0]
                    teacher_features = teacher_outputs[1] if len(teacher_outputs) > 1 else None
                else:
                    teacher_logits = teacher_outputs
                    teacher_features = None
                    
                # Apply temperature scaling
                soft_logits = teacher_logits / temperature
                soft_probs = F.softmax(soft_logits, dim=1)
                
                # Store targets
                soft_targets[f'task_{task_id}_logits'] = teacher_logits
                soft_targets[f'task_{task_id}_probs'] = soft_probs
                
                if teacher_features is not None:
                    soft_targets[f'task_{task_id}_features'] = teacher_features
                    
                # Extract gating information if available
                if hasattr(teacher, 'get_gating_outputs'):
                    gating_outputs = teacher.get_gating_outputs()
                    soft_targets[f'task_{task_id}_gating'] = gating_outputs
                    
        return soft_targets
        
    def update_temperatures(self, 
                           task_similarities: Dict[int, float],
                           current_task: int):
        """Update adaptive temperatures based on task similarities."""
        if not self.config.adaptive_temperature:
            return
            
        for task_id, similarity in task_similarities.items():
            if task_id in self.task_temperatures:
                # Higher similarity -> lower temperature (sharper targets)
                # Lower similarity -> higher temperature (softer targets)
                temperature_range = self.config.max_temperature - self.config.min_temperature
                new_temp = self.config.min_temperature + (1 - similarity) * temperature_range
                
                # Smooth update
                alpha = 0.7
                self.task_temperatures[task_id] = (
                    alpha * self.task_temperatures[task_id] + 
                    (1 - alpha) * new_temp
                )
                
    def get_teacher_predictions(self, 
                              inputs: torch.Tensor,
                              task_id: int,
                              device: torch.device) -> Optional[torch.Tensor]:
        """Get predictions from a specific teacher model."""
        if task_id not in self.teacher_models:
            return None
            
        teacher = self.teacher_models[task_id].to(device)
        
        with torch.no_grad():
            outputs = teacher(inputs.to(device))
            if isinstance(outputs, tuple):
                return outputs[0]
            return outputs


class FeatureDistiller:
    """
    Handles feature-level knowledge distillation between models.
    """
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.feature_adapters: nn.ModuleDict = nn.ModuleDict()
        
    def register_feature_layers(self, 
                              student_model: nn.Module,
                              teacher_model: nn.Module,
                              layer_pairs: List[Tuple[str, str]]):
        """
        Register feature layers for distillation.
        
        Args:
            student_model: Student model
            teacher_model: Teacher model  
            layer_pairs: List of (student_layer_name, teacher_layer_name) pairs
        """
        for student_layer, teacher_layer in layer_pairs:
            # Get layer dimensions
            student_module = dict(student_model.named_modules())[student_layer]
            teacher_module = dict(teacher_model.named_modules())[teacher_layer]
            
            # Create adapter if dimensions don't match
            if hasattr(student_module, 'out_features') and hasattr(teacher_module, 'out_features'):
                student_dim = student_module.out_features
                teacher_dim = teacher_module.out_features
                
                if student_dim != teacher_dim:
                    adapter = nn.Linear(student_dim, teacher_dim)
                    self.feature_adapters[f"{student_layer}_to_{teacher_layer}"] = adapter
                    
    def compute_feature_distillation_loss(self,
                                        student_features: Dict[str, torch.Tensor],
                                        teacher_features: Dict[str, torch.Tensor],
                                        device: torch.device) -> torch.Tensor:
        """Compute feature distillation loss."""
        total_loss = torch.tensor(0.0, device=device)
        num_losses = 0
        
        for layer_name in student_features:
            if layer_name in teacher_features:
                student_feat = student_features[layer_name]
                teacher_feat = teacher_features[layer_name]
                
                # Apply adapter if needed
                adapter_name = f"{layer_name}_adapter"
                if adapter_name in self.feature_adapters:
                    student_feat = self.feature_adapters[adapter_name](student_feat)
                    
                # Compute MSE loss between normalized features
                student_norm = F.normalize(student_feat, p=2, dim=1)
                teacher_norm = F.normalize(teacher_feat, p=2, dim=1)
                
                loss = F.mse_loss(student_norm, teacher_norm)
                total_loss += loss
                num_losses += 1
                
        return total_loss / max(1, num_losses)


class AttentionDistiller:
    """
    Handles attention-based knowledge distillation for MoE models.
    """
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        
    def compute_attention_distillation_loss(self,
                                          student_attention: torch.Tensor,
                                          teacher_attention: torch.Tensor,
                                          temperature: float = None) -> torch.Tensor:
        """
        Compute attention distillation loss.
        
        Args:
            student_attention: Student attention weights [batch_size, num_experts]
            teacher_attention: Teacher attention weights [batch_size, num_experts]
            temperature: Temperature for softmax (optional)
            
        Returns:
            Attention distillation loss
        """
        if temperature is None:
            temperature = self.config.temperature
            
        # Ensure same dimensions
        min_experts = min(student_attention.size(1), teacher_attention.size(1))
        student_attn = student_attention[:, :min_experts]
        teacher_attn = teacher_attention[:, :min_experts]
        
        # Apply temperature scaling
        student_soft = F.softmax(student_attn / temperature, dim=1)
        teacher_soft = F.softmax(teacher_attn / temperature, dim=1)
        
        # KL divergence loss
        loss = F.kl_div(
            F.log_softmax(student_attn / temperature, dim=1),
            teacher_soft,
            reduction='batchmean'
        )
        
        return loss * (temperature ** 2)  # Scale by temperature squared
        
    def compute_routing_consistency_loss(self,
                                       student_routing: Dict[str, torch.Tensor],
                                       teacher_routing: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for routing consistency between models."""
        total_loss = torch.tensor(0.0, device=next(iter(student_routing.values())).device)
        num_losses = 0
        
        for key in student_routing:
            if key in teacher_routing:
                student_route = student_routing[key]
                teacher_route = teacher_routing[key]
                
                # Compute similarity loss
                loss = 1.0 - F.cosine_similarity(
                    student_route.flatten(1),
                    teacher_route.flatten(1),
                    dim=1
                ).mean()
                
                total_loss += loss
                num_losses += 1
                
        return total_loss / max(1, num_losses)


class KnowledgeDistiller:
    """
    Main knowledge distillation coordinator.
    
    Combines output distillation, feature distillation, and attention distillation.
    """
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.soft_target_generator = SoftTargetGenerator(config)
        self.feature_distiller = FeatureDistiller(config)
        self.attention_distiller = AttentionDistiller(config)
        
    def register_teacher_model(self, 
                              task_id: int,
                              model: nn.Module,
                              class_mapping: Optional[Dict[int, int]] = None):
        """Register a teacher model for knowledge distillation."""
        self.soft_target_generator.register_teacher_model(task_id, model, class_mapping)
        
    def compute_distillation_loss(self,
                                student_outputs: torch.Tensor,
                                inputs: torch.Tensor,
                                current_task: int,
                                previous_tasks: List[int],
                                device: torch.device,
                                student_features: Optional[Dict[str, torch.Tensor]] = None,
                                student_attention: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive distillation loss.
        
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        if not previous_tasks:
            # No previous tasks to distill from
            return {'total_distillation_loss': torch.tensor(0.0, device=device)}
            
        # Generate soft targets
        soft_targets = self.soft_target_generator.generate_soft_targets(
            inputs, previous_tasks, device
        )
        
        # Output distillation loss
        output_loss = self._compute_output_distillation_loss(
            student_outputs, soft_targets, current_task, device
        )
        losses['output_distillation'] = output_loss
        
        # Feature distillation loss
        if self.config.use_feature_distillation and student_features:
            feature_loss = self._compute_feature_distillation_loss(
                student_features, soft_targets, device
            )
            losses['feature_distillation'] = feature_loss
        else:
            losses['feature_distillation'] = torch.tensor(0.0, device=device)
            
        # Attention distillation loss  
        if self.config.use_attention_distillation and student_attention:
            attention_loss = self._compute_attention_distillation_loss(
                student_attention, soft_targets, device
            )
            losses['attention_distillation'] = attention_loss
        else:
            losses['attention_distillation'] = torch.tensor(0.0, device=device)
            
        # Combine losses
        total_loss = (
            losses['output_distillation'] + 
            self.config.beta * losses['feature_distillation'] +
            self.config.beta * losses['attention_distillation']
        )
        
        losses['total_distillation_loss'] = total_loss
        
        return losses
        
    def _compute_output_distillation_loss(self,
                                        student_outputs: torch.Tensor,
                                        soft_targets: Dict[str, torch.Tensor],
                                        current_task: int,
                                        device: torch.device) -> torch.Tensor:
        """Compute output-level distillation loss."""
        total_loss = torch.tensor(0.0, device=device)
        num_losses = 0
        
        for key, targets in soft_targets.items():
            if key.endswith('_probs'):
                # Extract task ID from key
                task_id = int(key.split('_')[1])
                temperature = self.soft_target_generator.task_temperatures[task_id]
                
                # Compute KL divergence
                student_soft = F.log_softmax(student_outputs / temperature, dim=1)
                
                # Handle different class spaces
                if student_outputs.size(1) != targets.size(1):
                    # Align class dimensions (zero-pad or truncate)
                    min_classes = min(student_outputs.size(1), targets.size(1))
                    student_soft = student_soft[:, :min_classes]
                    targets = targets[:, :min_classes]
                    
                loss = F.kl_div(student_soft, targets, reduction='batchmean')
                loss *= (temperature ** 2)  # Scale by temperature squared
                
                total_loss += loss
                num_losses += 1
                
        return total_loss / max(1, num_losses)
        
    def _compute_feature_distillation_loss(self,
                                         student_features: Dict[str, torch.Tensor],
                                         soft_targets: Dict[str, torch.Tensor],
                                         device: torch.device) -> torch.Tensor:
        """Compute feature-level distillation loss."""
        total_loss = torch.tensor(0.0, device=device)
        num_losses = 0
        
        for key, teacher_features in soft_targets.items():
            if key.endswith('_features'):
                # Find corresponding student features
                layer_name = key.replace('_features', '')
                
                if layer_name in student_features:
                    loss = self.feature_distiller.compute_feature_distillation_loss(
                        {layer_name: student_features[layer_name]},
                        {layer_name: teacher_features},
                        device
                    )
                    total_loss += loss
                    num_losses += 1
                    
        return total_loss / max(1, num_losses)
        
    def _compute_attention_distillation_loss(self,
                                           student_attention: torch.Tensor,
                                           soft_targets: Dict[str, torch.Tensor],
                                           device: torch.device) -> torch.Tensor:
        """Compute attention distillation loss."""
        total_loss = torch.tensor(0.0, device=device)
        num_losses = 0
        
        for key, teacher_attention in soft_targets.items():
            if key.endswith('_gating'):
                loss = self.attention_distiller.compute_attention_distillation_loss(
                    student_attention, teacher_attention
                )
                total_loss += loss
                num_losses += 1
                
        return total_loss / max(1, num_losses)
        
    def update_task_similarities(self,
                               current_task: int,
                               task_similarities: Dict[int, float]):
        """Update temperature scaling based on task similarities."""
        self.soft_target_generator.update_temperatures(task_similarities, current_task)
        
    def get_distillation_info(self) -> Dict[str, Any]:
        """Get information about current distillation state."""
        return {
            'config': {
                'temperature': self.config.temperature,
                'alpha': self.config.alpha,
                'beta': self.config.beta,
                'use_feature_distillation': self.config.use_feature_distillation,
                'use_attention_distillation': self.config.use_attention_distillation,
            },
            'registered_teachers': list(self.soft_target_generator.teacher_models.keys()),
            'task_temperatures': dict(self.soft_target_generator.task_temperatures),
            'feature_adapters': list(self.feature_distiller.feature_adapters.keys())
        }