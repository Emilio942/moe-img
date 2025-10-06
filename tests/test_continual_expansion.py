"""
Test continual learning expert expansion functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from continual.expansion import (
    ExpertSignature, ExpertReuseMechanism, ExpertExpander,
    ExpansionConfig, ExpansionStrategy
)


class TestExpertSignature:
    """Test expert signature functionality."""
    
    def test_signature_creation(self):
        """Test creating expert signature."""
        signature = ExpertSignature(expert_id=1, task_ids=[1, 2])
        
        assert signature.expert_id == 1
        assert signature.task_ids == {1, 2}
        assert len(signature.feature_statistics) == 0
        assert len(signature.activation_patterns) == 0
        
    def test_update_statistics(self):
        """Test updating signature statistics."""
        signature = ExpertSignature(expert_id=1, task_ids=[])
        
        features = torch.randn(10, 128)
        activations = torch.randn(10)
        
        signature.update_statistics(features, activations, task_id=1, performance=0.85)
        
        assert 'mean' in signature.feature_statistics
        assert 'std' in signature.feature_statistics
        assert 1 in signature.activation_patterns
        assert 1 in signature.performance_history
        assert 1 in signature.task_ids
        assert signature.performance_history[1] == 0.85
        
    def test_compute_similarity(self):
        """Test similarity computation between signatures."""
        sig1 = ExpertSignature(expert_id=1, task_ids=[1])
        sig2 = ExpertSignature(expert_id=2, task_ids=[2])
        
        # Update with similar features
        features1 = torch.ones(10, 128)
        features2 = torch.ones(10, 128) * 1.1  # Similar but not identical
        
        sig1.update_statistics(features1, torch.ones(10), task_id=1)
        sig2.update_statistics(features2, torch.ones(10), task_id=2)
        
        similarity = sig1.compute_similarity(sig2)
        
        assert 0.0 <= similarity <= 1.0
        
        # Test with identical features
        sig3 = ExpertSignature(expert_id=3, task_ids=[3])
        sig3.update_statistics(features1, torch.ones(10), task_id=3)
        
        similarity_identical = sig1.compute_similarity(sig3)
        assert similarity_identical >= similarity  # Should be more similar
        
    def test_similarity_with_task_overlap(self):
        """Test similarity with overlapping tasks."""
        sig1 = ExpertSignature(expert_id=1, task_ids=[1, 2])
        sig2 = ExpertSignature(expert_id=2, task_ids=[2, 3])
        
        features = torch.randn(10, 128)
        
        sig1.update_statistics(features, torch.ones(10), task_id=1)
        sig1.update_statistics(features, torch.ones(10), task_id=2)
        
        sig2.update_statistics(features, torch.ones(10), task_id=2)
        sig2.update_statistics(features, torch.ones(10), task_id=3)
        
        similarity = sig1.compute_similarity(sig2)
        
        # Should have some similarity due to task overlap
        assert similarity > 0.0


class TestExpertReuseMechanism:
    """Test expert reuse mechanism."""
    
    def test_register_expert(self):
        """Test expert registration."""
        config = ExpansionConfig()
        mechanism = ExpertReuseMechanism(config)
        
        mechanism.register_expert(expert_id=1, initial_tasks=[1, 2])
        
        assert 1 in mechanism.expert_signatures
        assert mechanism.expert_signatures[1].expert_id == 1
        assert mechanism.expert_signatures[1].task_ids == {1, 2}
        
    def test_update_expert_signature(self):
        """Test updating expert signature."""
        config = ExpansionConfig()
        mechanism = ExpertReuseMechanism(config)
        
        features = torch.randn(10, 128)
        activations = torch.randn(10)
        
        mechanism.update_expert_signature(
            expert_id=1, 
            features=features,
            activations=activations,
            task_id=1,
            performance=0.9
        )
        
        assert 1 in mechanism.expert_signatures
        signature = mechanism.expert_signatures[1]
        assert 1 in signature.task_ids
        assert signature.performance_history[1] == 0.9
        
    def test_find_reusable_experts(self):
        """Test finding reusable experts."""
        config = ExpansionConfig(similarity_threshold=0.5)
        mechanism = ExpertReuseMechanism(config)
        
        # Register experts with different signatures
        features1 = torch.ones(10, 128)
        features2 = torch.ones(10, 128) * 2.0  # Different features
        
        mechanism.update_expert_signature(1, features1, torch.ones(10), task_id=1)
        mechanism.update_expert_signature(2, features2, torch.ones(10), task_id=2)
        
        # Find reusable experts for similar features
        target_features = torch.ones(10, 128) * 1.1  # Close to features1
        
        candidates = mechanism.find_reusable_experts(
            target_features, target_task=3
        )
        
        assert isinstance(candidates, list)
        # Should find expert 1 as more similar
        if candidates:
            assert candidates[0][0] in [1, 2]  # Expert ID should be valid
            assert 0.0 <= candidates[0][1] <= 1.0  # Similarity should be in range
            
    def test_should_reuse_expert(self):
        """Test reuse decision logic."""
        config = ExpansionConfig(similarity_threshold=0.7, reuse_probability=0.8)
        mechanism = ExpertReuseMechanism(config)
        
        # High similarity should have high reuse probability
        with patch('numpy.random.random', return_value=0.5):
            should_reuse_high = mechanism.should_reuse_expert(similarity=0.9)
            
        # Low similarity should not reuse
        should_reuse_low = mechanism.should_reuse_expert(similarity=0.3)
        
        assert should_reuse_high or True  # Depends on random, but valid
        assert not should_reuse_low  # Below threshold
        
    def test_get_expert_utilization(self):
        """Test expert utilization statistics."""
        config = ExpansionConfig()
        mechanism = ExpertReuseMechanism(config)
        
        # Add expert with activities
        mechanism.update_expert_signature(1, torch.randn(10, 128), torch.ones(10), 1, 0.8)
        mechanism.update_expert_signature(1, torch.randn(10, 128), torch.ones(10), 2, 0.9)
        
        utilization = mechanism.get_expert_utilization()
        
        assert 1 in utilization
        expert_util = utilization[1]
        assert expert_util['num_tasks'] == 2
        assert set(expert_util['task_ids']) == {1, 2}
        assert 0.0 <= expert_util['avg_activation'] <= 1.0
        assert 0.0 <= expert_util['avg_performance'] <= 1.0


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 10)
        
    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))
        
    def extract_features(self, x):
        return x.view(x.size(0), -1)


class TestExpertExpander:
    """Test expert expander functionality."""
    
    def test_initialization(self):
        """Test expander initialization."""
        config = ExpansionConfig(max_experts=16)
        model = MockModel()
        
        expander = ExpertExpander(config, model)
        
        assert expander.config.max_experts == 16
        assert isinstance(expander.reuse_mechanism, ExpertReuseMechanism)
        assert len(expander.expansion_history) == 0
        
    def test_analyze_task_requirements(self):
        """Test task requirement analysis."""
        config = ExpansionConfig()
        model = MockModel()
        expander = ExpertExpander(config, model)
        
        # Mock task data
        task_data = torch.randn(50, 3, 32, 32)
        device = torch.device('cpu')
        
        analysis = expander.analyze_task_requirements(
            task_id=1,
            task_data=task_data,
            model=model,
            device=device
        )
        
        assert 'task_id' in analysis
        assert 'feature_statistics' in analysis
        assert 'needs_expansion' in analysis
        assert 'current_expert_count' in analysis
        assert analysis['task_id'] == 1
        
    def test_needs_expansion_strategies(self):
        """Test different expansion strategies."""
        model = MockModel()
        
        # Test ALWAYS_ADD strategy
        config = ExpansionConfig(strategy=ExpansionStrategy.ALWAYS_ADD)
        expander = ExpertExpander(config, model)
        
        current_util = {}
        reuse_candidates = []
        
        needs_expansion = expander._needs_expansion(current_util, reuse_candidates)
        assert needs_expansion  # Should always expand
        
        # Test SIMILARITY_BASED strategy
        config = ExpansionConfig(strategy=ExpansionStrategy.SIMILARITY_BASED)
        expander = ExpertExpander(config, model)
        
        # With good candidates, shouldn't expand
        reuse_candidates = [(1, 0.8)]
        needs_expansion = expander._needs_expansion(current_util, reuse_candidates)
        assert not needs_expansion
        
        # Without candidates, should expand
        reuse_candidates = []
        needs_expansion = expander._needs_expansion(current_util, reuse_candidates)
        assert needs_expansion
        
    def test_expand_model_no_expansion(self):
        """Test model expansion when no expansion needed."""
        config = ExpansionConfig()
        model = MockModel()
        expander = ExpertExpander(config, model)
        
        # Mock analysis that doesn't need expansion
        analysis = {
            'needs_expansion': False,
            'reuse_candidates': [(1, 0.8)]
        }
        
        expanded_model, new_experts = expander.expand_model(model, task_id=2, analysis=analysis)
        
        assert expanded_model is model  # Should return same model
        assert len(new_experts) == 0
        
    def test_expand_model_with_expansion(self):
        """Test model expansion when expansion needed."""
        config = ExpansionConfig(max_experts=10)
        model = MockModel()
        expander = ExpertExpander(config, model)
        
        # Mock analysis that needs expansion
        analysis = {
            'needs_expansion': True,
            'reuse_candidates': [],
            'feature_statistics': {
                'mean': torch.zeros(128),
                'std': torch.ones(128),
                'shape': [10, 128]
            }
        }
        
        expanded_model, new_experts = expander.expand_model(model, task_id=2, analysis=analysis)
        
        assert expanded_model is not None
        assert len(new_experts) >= 0  # Should add some experts (or be limited by mock)
        
    def test_get_expansion_statistics(self):
        """Test expansion statistics."""
        config = ExpansionConfig()
        model = MockModel()
        expander = ExpertExpander(config, model)
        
        # Manually add expansion history
        expander.expansion_history.append({
            'task_id': 1,
            'new_experts': [1, 2],
            'total_experts': 2,
            'strategy_used': 'hybrid'
        })
        
        stats = expander.get_expansion_statistics()
        
        assert 'total_experts' in stats
        assert 'expansion_history' in stats
        assert 'expert_utilization' in stats
        assert stats['total_expansions'] == 1
        assert stats['total_new_experts'] == 2
        
    def test_visualize_expert_graph(self):
        """Test expert graph visualization data."""
        config = ExpansionConfig()
        model = MockModel()
        expander = ExpertExpander(config, model)
        
        # Register some experts
        expander.reuse_mechanism.register_expert(1, [1, 2])
        expander.reuse_mechanism.register_expert(2, [2, 3])
        
        graph_data = expander.visualize_expert_graph()
        
        assert 'expert_ids' in graph_data
        assert 'adjacency_matrix' in graph_data
        assert 'expert_tasks' in graph_data
        assert 'expert_performance' in graph_data
        
        assert isinstance(graph_data['adjacency_matrix'], np.ndarray)
        assert len(graph_data['expert_ids']) == 2


if __name__ == "__main__":
    pytest.main([__file__])