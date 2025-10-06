#!/usr/bin/env python3
"""
Phase 8 Adaptive Evaluation - Demo Script
=========================================

Demonstrates the comprehensive evaluation suite with a simple baseline model.
This script creates a dummy baseline model and runs the full evaluation pipeline
to showcase all Phase 8 capabilities.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.adaptive_runner import run_comprehensive_evaluation
from models.expert_graph import ExpertGraph


def create_baseline_model(num_classes: int = 10, 
                         num_experts: int = 4) -> nn.Module:
    """
    Create a simple baseline MoE model for demonstration.
    
    Args:
        num_classes: Number of output classes
        num_experts: Number of experts in MoE
        
    Returns:
        MoE model
    """
    try:
        # Try to use the project's ExpertGraph model
        model = ExpertGraph(
            input_dim=3 * 32 * 32,  # CIFAR-10 flattened
            hidden_dim=128,
            num_classes=num_classes,
            num_experts=num_experts,
            top_k=2,
            use_cooperation=False  # Baseline without cooperation
        )
        print(f"✅ Created ExpertGraph baseline model ({num_experts} experts)")
        return model
        
    except Exception as e:
        print(f"⚠️ Could not create ExpertGraph model: {e}")
        print("📝 Creating simple CNN baseline instead...")
        
        # Fallback: Simple CNN model
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d(1)
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, num_classes)
                )
                
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
                
        model = SimpleCNN(num_classes)
        print(f"✅ Created simple CNN baseline model")
        return model


def setup_demo_environment(workspace_root: Path):
    """
    Set up the demo environment with necessary directories and checkpoints.
    
    Args:
        workspace_root: Path to workspace root
    """
    print("🏗️ Setting up demo environment...")
    
    # Create necessary directories
    directories = [
        "data",
        "checkpoints", 
        "reports",
        "reports/adaptive_evaluation"
    ]
    
    for dir_name in directories:
        dir_path = workspace_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        
    print(f"📁 Created directories: {', '.join(directories)}")
    
    # Create demo model checkpoints
    model_configs = [
        ("baseline", "Simple baseline model without optimization"),
        ("expert_graph_best", "Expert graph model with cooperation"),
        ("compressed_best", "Compressed model for efficiency"),
    ]
    
    created_models = []
    
    for model_name, description in model_configs:
        checkpoint_path = workspace_root / "checkpoints" / f"{model_name}.ckpt"
        
        if not checkpoint_path.exists():
            print(f"📦 Creating {model_name} checkpoint...")
            
            # Create model variant based on name
            if "compressed" in model_name:
                # Smaller model for compression demo
                model = create_baseline_model(num_classes=10, num_experts=2)
            else:
                # Standard model
                model = create_baseline_model(num_classes=10, num_experts=4)
                
            # Add some random trained weights
            with torch.no_grad():
                for param in model.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.zeros_(param)
                        
            # Save checkpoint
            checkpoint = {
                'model': model,
                'description': description,
                'created_at': datetime.now().isoformat(),
                'demo': True
            }
            
            torch.save(checkpoint, checkpoint_path)
            created_models.append(model_name)
            print(f"✅ Saved {model_name} checkpoint")
        else:
            print(f"♻️ Using existing {model_name} checkpoint")
            
    print(f"🎯 Demo environment ready with {len(created_models)} new checkpoints")
    return created_models


def run_phase8_demo(workspace_root: str = ".",
                   quick_mode: bool = True,
                   seeds: list = None):
    """
    Run complete Phase 8 adaptive evaluation demo.
    
    Args:
        workspace_root: Path to workspace root
        quick_mode: Whether to run in quick mode (limited evaluation)
        seeds: Random seeds for evaluation
    """
    
    workspace_path = Path(workspace_root).resolve()
    
    if seeds is None:
        seeds = [17, 42, 1337]  # As specified in aufgabenliste
        
    print("🚀 Phase 8 Adaptive Evaluation Demo")
    print("=" * 50)
    print(f"📍 Workspace: {workspace_path}")
    print(f"🎲 Seeds: {seeds}")
    print(f"⚡ Quick mode: {quick_mode}")
    print()
    
    # Set up demo environment
    setup_demo_environment(workspace_path)
    
    print("\n🔄 Starting comprehensive adaptive evaluation...")
    print("-" * 50)
    
    try:
        # Run comprehensive evaluation
        results = run_comprehensive_evaluation(
            workspace_root=str(workspace_path),
            results_root=str(workspace_path / "reports" / "adaptive_evaluation"),
            seeds=seeds,
            quick_mode=quick_mode
        )
        
        print("\n🎉 Phase 8 Evaluation Completed Successfully!")
        print("=" * 50)
        
        # Print summary results
        summary = results.get('evaluation_summary', {})
        print(f"📊 Total experiments: {summary.get('total_experiments', 0)}")
        print(f"✅ Successful: {summary.get('successful_experiments', 0)}")
        print(f"❌ Failed: {len(summary.get('failed_experiments', []))}")
        print(f"⏱️ Duration: {summary.get('total_duration_seconds', 0):.1f} seconds")
        
        # Print key results
        if 'analysis_results' in results:
            analysis = results['analysis_results']
            
            if 'cost_efficiency' in analysis:
                cost_analysis = analysis['cost_efficiency']
                extracted_metrics = cost_analysis.get('extracted_metrics', {})
                print(f"\n💰 Cost Analysis: {len(extracted_metrics)} models analyzed")
                
            if 'calibration' in analysis:
                cal_analysis = analysis['calibration']
                cal_summary = cal_analysis.get('calibration_summary', {})
                if 'overall_calibration_quality' in cal_summary:
                    quality = cal_summary['overall_calibration_quality']
                    print(f"🎯 Calibration Quality: {quality.get('quality_assessment', 'Unknown')}")
                    
            if 'robustness' in analysis:
                rob_analysis = analysis['robustness']
                rob_summary = rob_analysis.get('summary', {})
                if rob_summary:
                    mean_drop = rob_summary.get('mean_accuracy_drop', 0.0)
                    print(f"🛡️ Mean Robustness Drop: {mean_drop:.1f}%")
                    
        # Print recommendations
        if 'recommendations' in results:
            recommendations = results['recommendations']
            
            print("\n💡 Recommendations:")
            
            for mode, mode_data in recommendations.items():
                if mode.endswith('_mode') and mode_data:
                    mode_name = mode.replace('_mode', '').replace('_', ' ').title()
                    recommended_model = mode_data.get('recommended_model', 'N/A')
                    print(f"  🎛️ {mode_name}: {recommended_model}")
                    
        # Print file locations
        print(f"\n📁 Results saved to:")
        results_dir = workspace_path / "reports" / "adaptive_evaluation"
        
        key_files = [
            "adaptive_evaluation_final_report.json",
            "adaptive_dashboard.md",
            "evaluation_summary.csv"
        ]
        
        for file_name in key_files:
            file_path = results_dir / file_name
            if file_path.exists():
                print(f"  📄 {file_path}")
                
        # Print dashboard location
        if 'dashboard_path' in results:
            print(f"\n📋 Dashboard: {results['dashboard_path']}")
            
        print("\n✨ Phase 8 Demo Complete! Check the generated reports and dashboard.")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point for demo script."""
    
    parser = argparse.ArgumentParser(description="Phase 8 Adaptive Evaluation Demo")
    parser.add_argument("--workspace", type=str, default=".", 
                       help="Workspace root directory (default: current directory)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 1337],
                       help="Random seeds for evaluation (default: 17 42 1337)")
    parser.add_argument("--full", action="store_true",
                       help="Run full evaluation (default: quick mode)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress some noisy loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Run demo
    quick_mode = not args.full
    
    results = run_phase8_demo(
        workspace_root=args.workspace,
        quick_mode=quick_mode,
        seeds=args.seeds
    )
    
    if results is None:
        sys.exit(1)
    else:
        # Check completion status
        completion_status = results.get('phase8_completion_status', {})
        completed_components = sum(1 for status in completion_status.values() if status)
        total_components = len(completion_status)
        completion_rate = (completed_components / max(total_components, 1)) * 100
        
        print(f"\n🎯 Phase 8 Completion: {completion_rate:.1f}% ({completed_components}/{total_components})")
        
        if completion_rate >= 80:
            print("🏆 Phase 8 requirements successfully met!")
            sys.exit(0)
        else:
            print("⚠️ Phase 8 partially complete - check logs for details")
            sys.exit(1)


if __name__ == "__main__":
    main()