"""
Expert expansion and dynamic graph modification for continual learning.

This module handles dynamic expansion of the expert graph when new tasks require
additional capacity, while respecting computational budgets and reusing existing experts.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import copy
import numpy as np
from collections import defaultdict


class ExpansionStrategy(Enum):
    """Strategy for expert expansion."""
    ALWAYS_ADD = "always_add"  # Always add new experts for new tasks
    SIMILARITY_BASED = "similarity"  # Reuse similar experts
    CAPACITY_BASED = "capacity"  # Add only when capacity is exceeded
    HYBRID = "hybrid"  # Combination of similarity and capacity


@dataclass
class ExpansionConfig:
    """Configuration for expert expansion."""
    strategy: ExpansionStrategy = ExpansionStrategy.HYBRID
    max_experts: int = 32  # Maximum number of experts
    similarity_threshold: float = 0.7  # Threshold for expert reuse
    capacity_threshold: float = 0.8  # Threshold for capacity-based expansion
    feature_dim: int = 512  # Dimension for feature similarity computation
    expert_hidden_dim: int = 256  # Hidden dimension for new experts
    reuse_probability: float = 0.5  # Probability of reusing vs adding new expert
    warmup_epochs: int = 5  # Epochs to warm up new experts


class ExpertSignature:
    """
    Represents the signature/fingerprint of an expert for similarity computation.
    """
    
    def __init__(self, expert_id: int, task_ids: List[int]):
        self.expert_id = expert_id
        self.task_ids = set(task_ids)
        self.feature_statistics: Dict[str, torch.Tensor] = {}
        self.activation_patterns: Dict[int, float] = {}  # task_id -> avg_activation
        self.performance_history: Dict[int, float] = {}  # task_id -> accuracy
        
    def update_statistics(self, 
                         features: torch.Tensor,
                         activations: torch.Tensor,
                         task_id: int,
                         performance: Optional[float] = None):
        """Update expert signature with new statistics."""
        # Update feature statistics
        self.feature_statistics['mean'] = features.mean(dim=0)
        self.feature_statistics['std'] = features.std(dim=0)
        
        # Update activation patterns
        self.activation_patterns[task_id] = float(activations.mean().item())
        
        # Update performance history
        if performance is not None:
            self.performance_history[task_id] = performance
            
        # Track tasks this expert has seen
        self.task_ids.add(task_id)
        
    def compute_similarity(self, other: 'ExpertSignature') -> float:
        """
        Compute similarity with another expert signature.
        
        Returns:
            Similarity score between 0 and 1
        """
        similarities = []
        
        # Feature similarity (cosine similarity of means)
        if ('mean' in self.feature_statistics and 
            'mean' in other.feature_statistics):
            mean1 = self.feature_statistics['mean']
            mean2 = other.feature_statistics['mean']
            
            cos_sim = torch.cosine_similarity(mean1, mean2, dim=0)
            similarities.append(float(cos_sim.item()))
            
        # Task overlap similarity
        if self.task_ids and other.task_ids:
            overlap = len(self.task_ids.intersection(other.task_ids))
            union = len(self.task_ids.union(other.task_ids))
            task_sim = overlap / union if union > 0 else 0.0
            similarities.append(task_sim)
            
        # Activation pattern similarity
        common_tasks = set(self.activation_patterns.keys()).intersection(
            set(other.activation_patterns.keys())
        )
        
        if common_tasks:
            act_corr = np.corrcoef(
                [self.activation_patterns[t] for t in common_tasks],
                [other.activation_patterns[t] for t in common_tasks]
            )[0, 1]
            
            if not np.isnan(act_corr):
                similarities.append(max(0.0, act_corr))
                
        return np.mean(similarities) if similarities else 0.0


class ExpertReuseMechanism:
    """
    Manages expert reuse decisions based on similarity and performance.
    """
    
    def __init__(self, config: ExpansionConfig):
        self.config = config
        self.expert_signatures: Dict[int, ExpertSignature] = {}
        self._next_expert_id = 0
        
    def register_expert(self, expert_id: int, initial_tasks: List[int] = None):
        """Register a new expert with initial task assignments."""
        if initial_tasks is None:
            initial_tasks = []
        self.expert_signatures[expert_id] = ExpertSignature(expert_id, initial_tasks)
        self._next_expert_id = max(self._next_expert_id, expert_id + 1)
        
    def update_expert_signature(self,
                              expert_id: int,
                              features: torch.Tensor,
                              activations: torch.Tensor, 
                              task_id: int,
                              performance: Optional[float] = None):
        """Update expert signature with new training data."""
        if expert_id not in self.expert_signatures:
            self.register_expert(expert_id, [task_id])
            
        self.expert_signatures[expert_id].update_statistics(
            features, activations, task_id, performance
        )
        
    def find_reusable_experts(self, 
                            target_features: torch.Tensor,
                            target_task: int,
                            exclude_experts: Set[int] = None) -> List[Tuple[int, float]]:
        """
        Find experts that can be reused for the target task.
        
        Returns:
            List of (expert_id, similarity_score) tuples, sorted by similarity
        """
        if exclude_experts is None:
            exclude_experts = set()
            
        candidates = []
        
        # Create temporary signature for target
        temp_signature = ExpertSignature(-1, [target_task])
        temp_signature.update_statistics(
            target_features, 
            torch.ones(target_features.size(0)), 
            target_task
        )
        
        for expert_id, signature in self.expert_signatures.items():
            if expert_id in exclude_experts:
                continue
                
            similarity = signature.compute_similarity(temp_signature)
            
            if similarity >= self.config.similarity_threshold:
                candidates.append((expert_id, similarity))
                
        # Sort by similarity (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates
        
    def should_reuse_expert(self, similarity: float, task_performance: float = None) -> bool:
        """
        Decide whether to reuse an expert based on similarity and performance.
        
        Args:
            similarity: Similarity score with candidate expert
            task_performance: Current task performance (if available)
            
        Returns:
            True if expert should be reused
        """
        if similarity < self.config.similarity_threshold:
            return False
            
        # Higher similarity increases reuse probability
        reuse_prob = self.config.reuse_probability * (similarity / self.config.similarity_threshold)
        
        # Performance-based adjustment
        if task_performance is not None:
            if task_performance > 0.8:  # Good performance
                reuse_prob *= 1.2
            elif task_performance < 0.5:  # Poor performance
                reuse_prob *= 0.8
                
        return np.random.random() < reuse_prob
        
    def get_expert_utilization(self) -> Dict[int, Dict[str, Any]]:
        """Get utilization statistics for all experts."""
        utilization = {}
        
        for expert_id, signature in self.expert_signatures.items():
            utilization[expert_id] = {
                'num_tasks': len(signature.task_ids),
                'task_ids': list(signature.task_ids),
                'avg_activation': np.mean(list(signature.activation_patterns.values())) 
                                if signature.activation_patterns else 0.0,
                'avg_performance': np.mean(list(signature.performance_history.values()))
                                 if signature.performance_history else 0.0
            }
            
        return utilization


class ExpertExpander:
    """
    Main class for managing expert graph expansion during continual learning.
    """
    
    def __init__(self, config: ExpansionConfig, initial_model: nn.Module):
        self.config = config
        self.reuse_mechanism = ExpertReuseMechanism(config)
        self.expansion_history: List[Dict[str, Any]] = []
        self._current_expert_count = 0
        self._task_expert_mapping: Dict[int, List[int]] = defaultdict(list)
        
        # Initialize with existing experts
        self._initialize_from_model(initial_model)
        
    def _initialize_from_model(self, model: nn.Module):
        """Initialize expert tracking from existing model."""
        # Count existing experts (assuming expert modules are named with 'expert')
        expert_count = 0
        for name, module in model.named_modules():
            if 'expert' in name.lower() and isinstance(module, nn.Module):
                expert_count += 1
                
        # Register existing experts
        for i in range(expert_count):
            self.reuse_mechanism.register_expert(i, [])
            
        self._current_expert_count = expert_count
        
    def analyze_task_requirements(self,
                                task_id: int,
                                task_data: torch.Tensor,
                                model: nn.Module,
                                device: torch.device) -> Dict[str, Any]:
        """
        Analyze requirements for a new task.
        
        Returns:
            Analysis results including capacity needs and reuse candidates
        """
        model.eval()
        
        with torch.no_grad():
            # Get features for task data
            sample_batch = task_data[:min(100, len(task_data))]  # Sample for analysis
            sample_batch = sample_batch.to(device)
            
            # Extract features (assuming model has a feature extraction method)
            if hasattr(model, 'extract_features'):
                features = model.extract_features(sample_batch)
            else:
                # Fallback: use output before final layer
                outputs = model(sample_batch)
                features = outputs  # Simplified assumption
                
        # Analyze current expert utilization
        current_utilization = self._analyze_current_utilization(model, sample_batch, device)
        
        # Find reusable experts
        reuse_candidates = self.reuse_mechanism.find_reusable_experts(
            features, task_id
        )
        
        # Determine expansion needs
        needs_expansion = self._needs_expansion(current_utilization, reuse_candidates)
        
        analysis = {
            'task_id': task_id,
            'feature_statistics': {
                'mean': features.mean(dim=0),
                'std': features.std(dim=0),
                'shape': list(features.shape)
            },
            'current_utilization': current_utilization,
            'reuse_candidates': reuse_candidates,
            'needs_expansion': needs_expansion,
            'current_expert_count': self._current_expert_count,
            'max_experts': self.config.max_experts
        }
        
        return analysis
        
    def _analyze_current_utilization(self,
                                   model: nn.Module,
                                   sample_data: torch.Tensor,
                                   device: torch.device) -> Dict[str, float]:
        """Analyze current expert utilization."""
        model.eval()
        
        utilization = {}
        expert_activations = {}
        
        # Hook to capture expert activations
        def capture_expert_output(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    expert_activations[name] = output.detach()
            return hook
            
        # Register hooks for expert modules
        hooks = []
        for name, module in model.named_modules():
            if 'expert' in name.lower():
                hook = module.register_forward_hook(capture_expert_output(name))
                hooks.append(hook)
                
        try:
            with torch.no_grad():
                _ = model(sample_data)
                
            # Compute utilization metrics
            for name, activation in expert_activations.items():
                utilization[name] = {
                    'mean_activation': float(activation.mean()),
                    'activation_std': float(activation.std()),
                    'sparsity': float((activation == 0).float().mean())
                }
                
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
                
        return utilization
        
    def _needs_expansion(self,
                       current_utilization: Dict[str, Any],
                       reuse_candidates: List[Tuple[int, float]]) -> bool:
        """Determine if expert expansion is needed."""
        if self._current_expert_count >= self.config.max_experts:
            return False  # Hit maximum capacity
            
        if self.config.strategy == ExpansionStrategy.ALWAYS_ADD:
            return True
            
        elif self.config.strategy == ExpansionStrategy.SIMILARITY_BASED:
            # Expand if no good reuse candidates
            return len(reuse_candidates) == 0
            
        elif self.config.strategy == ExpansionStrategy.CAPACITY_BASED:
            # Expand if current experts are highly utilized
            if not current_utilization:
                return True
                
            avg_utilization = np.mean([
                metrics['mean_activation'] for metrics in current_utilization.values()
                if isinstance(metrics, dict) and 'mean_activation' in metrics
            ])
            
            return avg_utilization > self.config.capacity_threshold
            
        elif self.config.strategy == ExpansionStrategy.HYBRID:
            # Combine similarity and capacity checks
            has_good_candidates = len(reuse_candidates) > 0
            
            if has_good_candidates:
                return False  # Reuse existing experts
                
            # Check capacity if no good candidates
            if current_utilization:
                avg_utilization = np.mean([
                    metrics['mean_activation'] for metrics in current_utilization.values()
                    if isinstance(metrics, dict) and 'mean_activation' in metrics
                ])
                return avg_utilization > self.config.capacity_threshold
                
            return True  # Default to expansion
            
        return False
        
    def expand_model(self,
                    model: nn.Module,
                    task_id: int,
                    analysis: Dict[str, Any]) -> Tuple[nn.Module, List[int]]:
        """
        Expand model with new experts for the given task.
        
        Returns:
            Tuple of (expanded_model, new_expert_ids)
        """
        new_expert_ids = []
        
        if not analysis['needs_expansion']:
            # Use existing experts
            reuse_candidates = analysis['reuse_candidates']
            if reuse_candidates:
                # Select best candidate for reuse
                best_expert_id, best_similarity = reuse_candidates[0]
                self._task_expert_mapping[task_id].append(best_expert_id)
                
                # Update expert signature
                features = analysis['feature_statistics']['mean']
                self.reuse_mechanism.update_expert_signature(
                    best_expert_id, 
                    features.unsqueeze(0), 
                    torch.tensor([best_similarity]),
                    task_id
                )
                
            return model, new_expert_ids
            
        # Create new expert(s)
        num_new_experts = self._determine_num_new_experts(analysis)
        
        for _ in range(num_new_experts):
            if self._current_expert_count >= self.config.max_experts:
                break
                
            new_expert_id = self._current_expert_count
            
            # Add new expert to model (this is model-specific)
            model = self._add_expert_to_model(model, new_expert_id)
            
            # Register new expert
            self.reuse_mechanism.register_expert(new_expert_id, [task_id])
            self._task_expert_mapping[task_id].append(new_expert_id)
            
            new_expert_ids.append(new_expert_id)
            self._current_expert_count += 1
            
        # Record expansion history
        self.expansion_history.append({
            'task_id': task_id,
            'new_experts': new_expert_ids,
            'total_experts': self._current_expert_count,
            'strategy_used': self.config.strategy.value,
            'analysis': analysis
        })
        
        return model, new_expert_ids
        
    def _determine_num_new_experts(self, analysis: Dict[str, Any]) -> int:
        """Determine number of new experts to add."""
        # Simple heuristic: add 1-2 experts per task
        base_experts = 1
        
        # Add more experts for complex tasks (high feature variance)
        feature_complexity = analysis['feature_statistics']['std'].mean().item()
        if feature_complexity > 0.5:
            base_experts += 1
            
        # Respect budget constraints
        remaining_budget = self.config.max_experts - self._current_expert_count
        return min(base_experts, remaining_budget)
        
    def _add_expert_to_model(self, model: nn.Module, expert_id: int) -> nn.Module:
        """
        Add a new expert to the model.
        
        This is a simplified version - actual implementation would depend on
        the specific model architecture.
        """
        # This would need to be implemented based on the specific MoE architecture
        # For now, return the model unchanged as a placeholder
        
        # In a real implementation, this would:
        # 1. Create new expert layers
        # 2. Expand the gating network
        # 3. Update routing mechanisms
        # 4. Initialize new parameters
        
        print(f"Would add expert {expert_id} to model (placeholder)")
        return model
        
    def get_task_experts(self, task_id: int) -> List[int]:
        """Get list of expert IDs assigned to a task."""
        return self._task_expert_mapping[task_id]
        
    def get_expansion_statistics(self) -> Dict[str, Any]:
        """Get statistics about expert expansion."""
        total_expansions = len(self.expansion_history)
        total_new_experts = sum(len(h['new_experts']) for h in self.expansion_history)
        
        utilization = self.reuse_mechanism.get_expert_utilization()
        
        return {
            'total_experts': self._current_expert_count,
            'max_experts': self.config.max_experts,
            'total_expansions': total_expansions,
            'total_new_experts': total_new_experts,
            'expansion_history': self.expansion_history,
            'expert_utilization': utilization,
            'task_expert_mapping': dict(self._task_expert_mapping)
        }
        
    def visualize_expert_graph(self) -> Dict[str, Any]:
        """Generate data for expert graph visualization."""
        # Create adjacency matrix based on task sharing
        expert_ids = list(self.reuse_mechanism.expert_signatures.keys())
        n_experts = len(expert_ids)
        
        adjacency = np.zeros((n_experts, n_experts))
        
        # Connect experts that share tasks
        for i, expert_i in enumerate(expert_ids):
            sig_i = self.reuse_mechanism.expert_signatures[expert_i]
            for j, expert_j in enumerate(expert_ids):
                if i != j:
                    sig_j = self.reuse_mechanism.expert_signatures[expert_j]
                    
                    # Edge weight based on task overlap
                    overlap = len(sig_i.task_ids.intersection(sig_j.task_ids))
                    union = len(sig_i.task_ids.union(sig_j.task_ids))
                    
                    if union > 0:
                        adjacency[i, j] = overlap / union
                        
        return {
            'expert_ids': expert_ids,
            'adjacency_matrix': adjacency,
            'expert_tasks': {
                expert_id: list(sig.task_ids) 
                for expert_id, sig in self.reuse_mechanism.expert_signatures.items()
            },
            'expert_performance': {
                expert_id: sig.performance_history
                for expert_id, sig in self.reuse_mechanism.expert_signatures.items()
            }
        }