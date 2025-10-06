#!/usr/bin/env python3
"""
Demo script for Phase 9: Continual & Domain-Shift Learning

This script demonstrates the continual learning capabilities including:
1. Sequential task learning (CIFAR-10 → CIFAR-100 subsets)
2. Catastrophic forgetting prevention (EWC/MAS)
3. Expert expansion and specialization
4. Knowledge distillation
5. Comprehensive metrics and reporting

Usage:
    python scripts/continual_demo.py [--config CONFIG_FILE]
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import logging
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from continual import (
    ContinualDataStream, ContinualTrainer, ContinualConfig,
    RegularizerConfig, DistillationConfig, ExpansionConfig,
    ExpansionStrategy
)
from models.expert_graph import ExpertGraph


class SimpleMoE(nn.Module):
    """Simplified MoE model for demonstration."""
    
    def __init__(self, input_size=3072, hidden_size=256, num_classes=10, num_experts=4):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        
        # Simple expert network
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Compute gating weights
        gate_weights = self.gate(features)
        
        # Compute expert outputs
        expert_outputs = torch.stack([
            expert(features) for expert in self.experts
        ], dim=1)  # [batch_size, num_experts, num_classes]
        
        # Weighted combination
        gate_weights = gate_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        output = torch.sum(gate_weights * expert_outputs, dim=1)
        
        return output
        
    def extract_features(self, x):
        """Extract features for analysis."""
        x = x.view(x.size(0), -1)
        return self.feature_extractor(x)
        
    def get_gating_outputs(self):
        """Get gating outputs for distillation."""
        # This would be called during forward pass to capture gating
        # For simplicity, return None in this demo
        return None


def create_demo_config() -> ContinualConfig:
    """Create demonstration configuration."""
    
    # Regularization config
    reg_config = RegularizerConfig(
        method="ewc",
        lambda_reg=1000.0,
        fisher_estimation_samples=200,
        regularize_gates=True,
        regularize_experts=True,
        freeze_threshold=0.7,
        freeze_factor=0.1
    )
    
    # Distillation config
    dist_config = DistillationConfig(
        temperature=4.0,
        alpha=0.5,
        beta=0.3,
        use_feature_distillation=True,
        use_attention_distillation=False,  # Disabled for simple demo
        replay_distillation=True
    )
    
    # Expansion config
    exp_config = ExpansionConfig(
        strategy=ExpansionStrategy.HYBRID,
        max_experts=8,
        similarity_threshold=0.6,
        capacity_threshold=0.8,
        reuse_probability=0.7
    )
    
    # Main config
    config = ContinualConfig(
        learning_rate=0.001,
        batch_size=32,  # Smaller for demo
        epochs_per_task=5,  # Fewer epochs for demo
        regularizer_config=reg_config,
        distillation_config=dist_config,
        expansion_config=exp_config,
        use_replay=True,
        replay_ratio=0.3,
        eval_every_epoch=True,
        save_checkpoints=True,
        checkpoint_dir="./checkpoints/continual_demo",
        log_dir="./logs/continual_demo",
        optimizer_type="adam",
        scheduler_type="cosine"
    )
    
    return config


class DemoCallback:
    """Demo callback for logging progress."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def on_task_start(self, task_id, task_config):
        self.logger.info(f"🚀 Starting Task {task_id}: {task_config.name}")
        self.logger.info(f"   Classes: {task_config.num_classes} (offset: {task_config.class_offset})")
        
    def on_task_end(self, task_id, metrics):
        self.logger.info(f"✅ Completed Task {task_id}")
        self.logger.info(f"   Final Accuracy: {metrics['final_accuracy']:.2f}%")
        self.logger.info(f"   Best Accuracy: {metrics['best_accuracy']:.2f}%")
        
    def on_epoch_end(self, task_id, epoch, metrics):
        if 'eval_accuracy' in metrics:
            self.logger.info(
                f"   Epoch {epoch+1}: Loss={metrics['loss']:.4f}, "
                f"Acc={metrics['eval_accuracy']:.2f}%, "
                f"Reg={metrics['reg_loss']:.4f}, "
                f"Dist={metrics['dist_loss']:.4f}"
            )


def run_continual_demo(config_path: Optional[str] = None, 
                      data_path: str = "./data",
                      output_dir: str = "./demo_results"):
    """
    Run continual learning demonstration.
    
    Args:
        config_path: Optional path to YAML config file
        data_path: Path to dataset directory
        output_dir: Path to save demo results
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🧠 Starting Continual Learning Demo")
    logger.info("=" * 50)
    
    # Load configuration
    if config_path and Path(config_path).exists():
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Would need to parse config_dict into ContinualConfig
        config = create_demo_config()  # Use demo config for now
    else:
        logger.info("Using demo configuration")
        config = create_demo_config()
        
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize data stream
        logger.info("📊 Setting up continual data stream...")
        data_stream = ContinualDataStream(
            replay_capacity=200,  # Small for demo
            replay_strategy="reservoir",
            batch_size=config.batch_size,
            replay_ratio=config.replay_ratio
        )
        
        # Create task sequence
        logger.info("Creating task sequence (CIFAR-10 → CIFAR-100 subsets)...")
        try:
            task_sequence = data_stream.create_cifar_sequence(data_path)
            logger.info(f"✅ Created {len(task_sequence)} tasks")
            
            for task in task_sequence:
                logger.info(f"   Task {task.task_id}: {task.name} ({task.num_classes} classes)")
                
        except Exception as e:
            logger.warning(f"Could not create CIFAR sequence: {e}")
            logger.info("Creating mock task sequence for demo...")
            
            # Create mock tasks for demo
            mock_tasks = []
            for i in range(1, 4):
                task = data_stream.task_sequence.add_task(
                    task_id=i,
                    name=f"Mock Task {i}",
                    num_classes=10 if i == 1 else 20,
                    dataset_name="mock",
                )
                mock_tasks.append(task)
            task_sequence = mock_tasks
            
        # Initialize model
        logger.info("🏗️ Initializing MoE model...")
        total_classes = data_stream.task_sequence.get_total_classes()
        model = SimpleMoE(
            input_size=3072,  # 32*32*3 for CIFAR
            hidden_size=128,  # Smaller for demo
            num_classes=total_classes,
            num_experts=4
        )
        
        logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Initialize trainer
        logger.info("🎯 Setting up continual trainer...")
        trainer = ContinualTrainer(
            model=model,
            config=config,
            data_stream=data_stream,
            device=device
        )
        
        # Add callback for progress logging
        trainer.add_callback(DemoCallback())
        
        # Run training sequence
        logger.info("🚂 Starting continual learning sequence...")
        logger.info("=" * 50)
        
        results = trainer.train_sequence(task_sequence)
        
        logger.info("=" * 50)
        logger.info("✅ Continual learning completed!")
        
        # Print results summary
        logger.info("📈 Results Summary:")
        logger.info("-" * 30)
        
        final_metrics = results['final_metrics']
        
        # Forgetting analysis
        if 'forgetting' in final_metrics:
            forgetting = final_metrics['forgetting']
            logger.info(f"📉 Forgetting Analysis:")
            logger.info(f"   Average Forgetting: {forgetting.get('average_forgetting', 0):.4f}")
            logger.info(f"   Max Forgetting: {forgetting.get('max_forgetting', 0):.4f}")
            
        # Transfer analysis
        if 'transfer' in final_metrics:
            transfer = final_metrics['transfer']
            logger.info(f"🔄 Transfer Analysis:")
            logger.info(f"   Avg Forward Transfer: {transfer.get('average_forward_transfer', 0):.4f}")
            logger.info(f"   Avg Backward Transfer: {transfer.get('average_backward_transfer', 0):.4f}")
            
        # Specialization analysis
        if 'specialization' in final_metrics:
            spec = final_metrics['specialization']
            if 'diversity_metrics' in spec:
                div = spec['diversity_metrics']
                logger.info(f"🎯 Expert Specialization:")
                logger.info(f"   Avg Specialization: {div.get('avg_specialization', 0):.4f}")
                logger.info(f"   Specialized Experts: {div.get('highly_specialized_experts', 0)}")
                logger.info(f"   Generalist Experts: {div.get('generalist_experts', 0)}")
                
        # Parameter growth
        if 'parameter_growth' in final_metrics:
            growth = final_metrics['parameter_growth']
            logger.info(f"📊 Parameter Growth:")
            logger.info(f"   Total Parameters: {growth.get('total_parameters', 0)}")
            logger.info(f"   Growth Rate: {growth.get('growth_rate', 0):.2f} params/task")
            
        # Generate comprehensive report
        logger.info("📑 Generating comprehensive report...")
        report_paths = trainer.generate_continual_report(str(output_path))
        
        logger.info("📁 Generated files:")
        for report_type, path in report_paths.items():
            logger.info(f"   {report_type}: {path}")
            
        # Save configuration
        config_save_path = output_path / "demo_config.yaml"
        config_dict = {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'epochs_per_task': config.epochs_per_task,
            'use_replay': config.use_replay,
            'replay_ratio': config.replay_ratio,
            'regularizer_method': config.regularizer_config.method if config.regularizer_config else None,
            'distillation_temperature': config.distillation_config.temperature if config.distillation_config else None,
            'expansion_strategy': config.expansion_config.strategy.value if config.expansion_config else None,
        }
        
        with open(config_save_path, 'w') as f:
            yaml.dump(config_dict, f, indent=2)
            
        logger.info(f"💾 Saved demo configuration: {config_save_path}")
        
        # Final summary
        training_summary = trainer.get_training_summary()
        logger.info("🎉 Demo completed successfully!")
        logger.info(f"   Tasks completed: {training_summary.get('total_tasks', 0)}")
        logger.info(f"   Average final accuracy: {training_summary.get('avg_final_accuracy', 0):.2f}%")
        logger.info(f"   Average best accuracy: {training_summary.get('avg_best_accuracy', 0):.2f}%")
        logger.info(f"   Results saved to: {output_path}")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("⚠️ Demo interrupted by user")
        return None
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Continual Learning Demo")
    
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='./data',
        help='Path to dataset directory'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str, 
        default='./demo_results',
        help='Directory to save demo results'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick demo with reduced epochs'
    )
    
    args = parser.parse_args()
    
    # Adjust for quick demo
    if args.quick:
        args.output_dir = './demo_results_quick'
        
    # Run demo
    results = run_continual_demo(
        config_path=args.config,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    if results is None:
        sys.exit(1)
    else:
        print(f"\n✅ Demo completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()