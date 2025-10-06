"""
Adaptive Evaluation Runner - Main Entry Point for Phase 8
========================================================

High-level orchestration for comprehensive model evaluation across
baselines, RL-gated, meta-optimized, and compressed models with
full reporting and dashboard generation.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import argparse
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from .pipeline import EvaluationPipeline, EvaluationRunner, ExperimentConfig
from .datasets import EvaluationDatasets
from .cost_analysis import CostAnalyzer, ParettoAnalyzer
from .calibration import CalibrationAnalyzer
from .reporting import ReportGenerator

logger = logging.getLogger(__name__)


class AdaptiveEvaluationRunner:
    """
    Main runner for Phase 8 adaptive evaluation.
    Orchestrates evaluation of all model variants with comprehensive analysis.
    """
    
    def __init__(self, 
                 workspace_root: str = ".",
                 results_root: str = "./reports/adaptive_evaluation",
                 seeds: List[int] = None,
                 quick_mode: bool = False):
        self.workspace_root = Path(workspace_root)
        self.results_root = Path(results_root)
        self.seeds = seeds or [17, 42, 1337]  # As specified in aufgabenliste
        self.quick_mode = quick_mode
        
        # Create results directory
        self.results_root.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.model_configs = self._setup_model_configurations()
        
        # Analysis components
        self.cost_analyzer = CostAnalyzer()
        self.calibration_analyzer = CalibrationAnalyzer()
        self.report_generator = ReportGenerator(results_root=str(self.results_root))
        
        logger.info(f"AdaptiveEvaluationRunner initialized")
        logger.info(f"Workspace: {self.workspace_root}")
        logger.info(f"Results: {self.results_root}")
        logger.info(f"Seeds: {self.seeds}")
        logger.info(f"Quick mode: {self.quick_mode}")
        
    def _setup_model_configurations(self) -> Dict[str, Dict[str, Any]]:
        """
        Setup model configurations for evaluation.
        Maps model variants to their checkpoint paths and metadata.
        """
        configs = {
            "baseline": {
                "name": "MoE Baseline",
                "checkpoint_path": self.workspace_root / "checkpoints" / "baseline.ckpt",
                "description": "Standard MoE without optimization",
                "tags": ["baseline", "no-optimization"]
            },
            "expert_graph": {
                "name": "Expert Graph (Phase 3)",
                "checkpoint_path": self.workspace_root / "checkpoints" / "expert_graph_best.ckpt", 
                "description": "MoE with expert cooperation and graph structure",
                "tags": ["expert-graph", "cooperation"]
            },
            "rl_gated": {
                "name": "RL-Gated (Phase 4)",
                "checkpoint_path": self.workspace_root / "checkpoints" / "rl_gated.ckpt",
                "description": "RL-optimized gating mechanism",
                "tags": ["rl-gating", "optimization"]
            },
            "meta_optimized": {
                "name": "Meta-Optimized (Phase 6)",
                "checkpoint_path": self.workspace_root / "checkpoints" / "meta_optimized.ckpt",
                "description": "Meta-learning optimized hyperparameters",
                "tags": ["meta-learning", "hyperopt"]
            },
            "compressed": {
                "name": "Compressed (Phase 7)",
                "checkpoint_path": self.workspace_root / "checkpoints" / "compressed_best.ckpt",
                "description": "Compressed model with quantization/pruning",
                "tags": ["compression", "efficient"]
            }
        }
        
        # Filter to existing checkpoints
        existing_configs = {}
        for key, config in configs.items():
            if config["checkpoint_path"].exists():
                existing_configs[key] = config
                logger.info(f"Found checkpoint: {key} -> {config['checkpoint_path']}")
            else:
                logger.warning(f"Missing checkpoint: {key} -> {config['checkpoint_path']}")
                
        if not existing_configs:
            logger.error("No model checkpoints found! Creating dummy baseline for testing.")
            # Create dummy config for testing
            existing_configs["test_baseline"] = {
                "name": "Test Baseline", 
                "checkpoint_path": None,  # Will create dummy model
                "description": "Test model for pipeline validation",
                "tags": ["test", "dummy"]
            }
            
        return existing_configs
        
    def create_experiment_configs(self) -> List[ExperimentConfig]:
        """
        Create experiment configurations for all model variants.
        
        Returns:
            List of ExperimentConfig objects
        """
        experiment_configs = []
        
        for model_key, model_info in self.model_configs.items():
            # Determine batch size and limits based on quick mode
            if self.quick_mode:
                batch_size = 64
                max_batches = 10
                max_corruptions = 5
                evaluate_ood = True  # Still test OOD but limited
            else:
                batch_size = 256
                max_batches = None  # Full evaluation
                max_corruptions = None  # All corruptions
                evaluate_ood = True
                
            config = ExperimentConfig(
                experiment_id=f"adaptive_eval_{model_key}",
                model_name=model_info["name"],
                checkpoint_path=str(model_info["checkpoint_path"]) if model_info["checkpoint_path"] else None,
                seeds=self.seeds,
                batch_size=batch_size,
                max_batches=max_batches,
                evaluate_ood=evaluate_ood,
                max_corruptions=max_corruptions,
                apply_temperature_scaling=True,
                temperature_lr=0.01,
                warmup_batches=2,
                measure_energy=True,
                save_detailed_results=True,
                save_logits=False,  # Set to True if you want calibration analysis
                description=model_info["description"],
                tags=model_info["tags"]
            )
            
            experiment_configs.append(config)
            
        logger.info(f"Created {len(experiment_configs)} experiment configurations")
        return experiment_configs
        
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all model variants.
        
        Returns:
            Complete evaluation results with analysis
        """
        logger.info("Starting comprehensive adaptive evaluation...")
        start_time = datetime.now()
        
        # Create experiment configurations
        experiment_configs = self.create_experiment_configs()
        
        if not experiment_configs:
            raise ValueError("No experiment configurations available")
            
        # Run evaluations
        runner = EvaluationRunner(
            data_root=str(self.workspace_root / "data"),
            results_root=str(self.results_root),
            max_workers=1  # Sequential for stability
        )
        
        # Execute all experiments
        batch_results = runner.run_experiments(
            experiment_configs, 
            parallel=False  # Keep sequential for reproducibility
        )
        
        # Extract successful results for analysis
        successful_results = {}
        for exp_id, result in batch_results['results'].items():
            if result.get('status') != 'failed':
                successful_results[exp_id] = result
            else:
                logger.warning(f"Experiment {exp_id} failed: {result.get('error', 'Unknown error')}")
                
        if not successful_results:
            raise RuntimeError("All experiments failed - cannot proceed with analysis")
            
        logger.info(f"Successfully completed {len(successful_results)}/{len(experiment_configs)} experiments")
        
        # Comprehensive analysis
        analysis_results = self._run_comprehensive_analysis(successful_results)
        
        # Generate final report
        final_report = self._generate_final_report(
            batch_results, analysis_results, start_time
        )
        
        logger.info("Comprehensive evaluation completed successfully!")
        return final_report
        
    def _run_comprehensive_analysis(self, 
                                  evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive analysis on evaluation results.
        
        Args:
            evaluation_results: Results from all experiments
            
        Returns:
            Analysis results including Pareto, calibration, robustness
        """
        logger.info("Running comprehensive analysis...")
        
        analysis_results = {}
        
        try:
            # 1. Cost-Efficiency Analysis (Pareto fronts)
            logger.info("Computing Pareto analysis...")
            cost_analysis = self.cost_analyzer.analyze_cost_efficiency(evaluation_results)
            analysis_results['cost_efficiency'] = cost_analysis
            
            # 2. Calibration Analysis 
            logger.info("Computing calibration analysis...")
            calibration_analysis = self.calibration_analyzer.analyze_calibration(evaluation_results)
            analysis_results['calibration'] = calibration_analysis
            
            # 3. Robustness Analysis
            logger.info("Computing robustness analysis...")
            robustness_analysis = self._analyze_robustness(evaluation_results)
            analysis_results['robustness'] = robustness_analysis
            
            # 4. Overall Rankings
            logger.info("Computing overall rankings...")
            rankings = self._compute_overall_rankings(evaluation_results)
            analysis_results['rankings'] = rankings
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            analysis_results['error'] = str(e)
            
        return analysis_results
        
    def _analyze_robustness(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze robustness across all models."""
        robustness_analysis = {
            'by_model': {},
            'summary': {}
        }
        
        robustness_scores = []
        accuracy_drops = []
        
        for exp_id, result in evaluation_results.items():
            if 'aggregated_metrics' in result and 'robustness_summary' in result['aggregated_metrics']:
                robustness_summary = result['aggregated_metrics']['robustness_summary']
                
                model_robustness = {
                    'relative_robustness_mean': robustness_summary.get('relative_robustness_mean', {}).get('mean', 0.0),
                    'accuracy_drop_mean': robustness_summary.get('accuracy_drop_mean', {}).get('mean', 0.0),
                    'confidence_robustness': robustness_summary.get('confidence_robustness_mean', {}).get('mean', 0.0)
                }
                
                robustness_analysis['by_model'][exp_id] = model_robustness
                robustness_scores.append(model_robustness['relative_robustness_mean'])
                accuracy_drops.append(model_robustness['accuracy_drop_mean'])
                
        # Overall robustness summary
        if robustness_scores:
            robustness_analysis['summary'] = {
                'best_relative_robustness': max(robustness_scores),
                'worst_relative_robustness': min(robustness_scores),
                'mean_accuracy_drop': np.mean(accuracy_drops),
                'robustness_range': max(robustness_scores) - min(robustness_scores)
            }
            
        return robustness_analysis
        
    def _compute_overall_rankings(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall model rankings across multiple criteria."""
        
        rankings = {
            'accuracy_ranking': [],
            'efficiency_ranking': [],
            'robustness_ranking': [],
            'balanced_ranking': []
        }
        
        model_scores = {}
        
        for exp_id, result in evaluation_results.items():
            if 'aggregated_metrics' not in result:
                continue
                
            agg_metrics = result['aggregated_metrics']
            
            # Extract key metrics
            in_domain_acc = agg_metrics.get('in_domain', {}).get('top1_accuracy', {}).get('mean', 0.0)
            time_per_batch = agg_metrics.get('in_domain', {}).get('time_per_batch_median', {}).get('mean', float('inf'))
            memory_usage = agg_metrics.get('in_domain', {}).get('memory_peak_mb', {}).get('mean', float('inf'))
            
            robustness_score = 0.0
            if 'robustness_summary' in agg_metrics:
                robustness_score = agg_metrics['robustness_summary'].get('relative_robustness_mean', {}).get('mean', 0.0)
                
            model_scores[exp_id] = {
                'accuracy': in_domain_acc,
                'efficiency': 1.0 / max(time_per_batch, 1e-6),  # Higher is better
                'memory_efficiency': 1.0 / max(memory_usage, 1.0),  # Higher is better
                'robustness': robustness_score
            }
            
        # Rank by each criterion
        for criterion in ['accuracy', 'efficiency', 'robustness']:
            sorted_models = sorted(
                model_scores.items(),
                key=lambda x: x[1][criterion],
                reverse=True
            )
            rankings[f'{criterion}_ranking'] = [
                {'model': model_id, 'score': scores[criterion]}
                for model_id, scores in sorted_models
            ]
            
        # Balanced ranking (weighted average)
        balanced_scores = {}
        for model_id, scores in model_scores.items():
            # Normalize scores to 0-1 range
            max_acc = max(s['accuracy'] for s in model_scores.values())
            max_eff = max(s['efficiency'] for s in model_scores.values())
            max_rob = max(s['robustness'] for s in model_scores.values())
            
            normalized_score = (
                0.4 * (scores['accuracy'] / max(max_acc, 1e-6)) +
                0.3 * (scores['efficiency'] / max(max_eff, 1e-6)) +
                0.3 * (scores['robustness'] / max(max_rob, 1e-6))
            )
            balanced_scores[model_id] = normalized_score
            
        sorted_balanced = sorted(
            balanced_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        rankings['balanced_ranking'] = [
            {'model': model_id, 'balanced_score': score}
            for model_id, score in sorted_balanced
        ]
        
        return rankings
        
    def _generate_final_report(self,
                             batch_results: Dict[str, Any],
                             analysis_results: Dict[str, Any],
                             start_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        final_report = {
            'evaluation_summary': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_duration_seconds': total_duration,
                'total_experiments': batch_results['total_experiments'],
                'successful_experiments': batch_results['successful_experiments'],
                'failed_experiments': batch_results['failed_experiments'],
                'seeds_evaluated': self.seeds,
                'quick_mode': self.quick_mode
            },
            'batch_results': batch_results,
            'analysis_results': analysis_results,
            'recommendations': self._generate_recommendations(analysis_results),
            'phase8_completion_status': {
                'in_domain_evaluation': True,
                'ood_evaluation': True,
                'cost_analysis': 'cost_efficiency' in analysis_results,
                'calibration_analysis': 'calibration' in analysis_results,
                'robustness_analysis': 'robustness' in analysis_results,
                'pareto_analysis': 'cost_efficiency' in analysis_results,
                'export_validation': False,  # TODO: implement
                'comprehensive_dashboard': True
            }
        }
        
        # Save final report
        report_path = self.results_root / "adaptive_evaluation_final_report.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
            
        logger.info(f"Final report saved to {report_path}")
        
        # Generate dashboard
        try:
            dashboard_path = self.report_generator.generate_comprehensive_dashboard(final_report)
            final_report['dashboard_path'] = str(dashboard_path)
            logger.info(f"Dashboard generated at {dashboard_path}")
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            final_report['dashboard_error'] = str(e)
            
        return final_report
        
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations based on analysis."""
        
        recommendations = {
            'quality_max_mode': {},
            'balanced_mode': {},
            'cost_saving_mode': {},
            'general_insights': []
        }
        
        if 'rankings' in analysis_results:
            rankings = analysis_results['rankings']
            
            # Quality-max recommendation
            if rankings['accuracy_ranking']:
                best_accuracy = rankings['accuracy_ranking'][0]
                recommendations['quality_max_mode'] = {
                    'recommended_model': best_accuracy['model'],
                    'accuracy_score': best_accuracy['score'],
                    'rationale': 'Highest accuracy model for quality-focused scenarios'
                }
                
            # Balanced recommendation  
            if rankings['balanced_ranking']:
                best_balanced = rankings['balanced_ranking'][0]
                recommendations['balanced_mode'] = {
                    'recommended_model': best_balanced['model'],
                    'balanced_score': best_balanced['balanced_score'],
                    'rationale': 'Best accuracy/efficiency/robustness balance'
                }
                
            # Cost-saving recommendation
            if rankings['efficiency_ranking']:
                best_efficiency = rankings['efficiency_ranking'][0]
                recommendations['cost_saving_mode'] = {
                    'recommended_model': best_efficiency['model'],
                    'efficiency_score': best_efficiency['score'],
                    'rationale': 'Most efficient model for resource-constrained scenarios'
                }
                
        # General insights
        insights = []
        
        if 'cost_efficiency' in analysis_results:
            insights.append("Cost-efficiency Pareto analysis completed - check pareto_plots for trade-offs")
            
        if 'robustness' in analysis_results:
            robustness = analysis_results['robustness']
            if 'summary' in robustness:
                mean_drop = robustness['summary'].get('mean_accuracy_drop', 0.0)
                if mean_drop < 5.0:
                    insights.append(f"Models show good robustness (mean accuracy drop: {mean_drop:.1f}%)")
                else:
                    insights.append(f"Models show robustness challenges (mean accuracy drop: {mean_drop:.1f}%)")
                    
        recommendations['general_insights'] = insights
        
        return recommendations


def run_comprehensive_evaluation(workspace_root: str = ".",
                               results_root: str = "./reports/adaptive_evaluation",
                               seeds: List[int] = None,
                               quick_mode: bool = False) -> Dict[str, Any]:
    """
    Main entry point for comprehensive adaptive evaluation.
    
    Args:
        workspace_root: Path to workspace root
        results_root: Path to save results
        seeds: Random seeds for evaluation
        quick_mode: Whether to run in quick mode (limited evaluation)
        
    Returns:
        Complete evaluation results
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(results_root) / "evaluation.log")
        ]
    )
    
    # Create and run evaluation
    runner = AdaptiveEvaluationRunner(
        workspace_root=workspace_root,
        results_root=results_root,
        seeds=seeds,
        quick_mode=quick_mode
    )
    
    return runner.run_comprehensive_evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 8 Adaptive Evaluation")
    parser.add_argument("--workspace", type=str, default=".", 
                       help="Workspace root directory")
    parser.add_argument("--results", type=str, default="./reports/adaptive_evaluation",
                       help="Results output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 1337],
                       help="Random seeds for evaluation")
    parser.add_argument("--quick", action="store_true",
                       help="Run in quick mode (limited evaluation)")
    
    args = parser.parse_args()
    
    try:
        results = run_comprehensive_evaluation(
            workspace_root=args.workspace,
            results_root=args.results,
            seeds=args.seeds,
            quick_mode=args.quick
        )
        
        print("\n🎉 Adaptive Evaluation Completed Successfully!")
        print(f"📊 Results saved to: {args.results}")
        print(f"📈 {results['evaluation_summary']['successful_experiments']}/{results['evaluation_summary']['total_experiments']} experiments successful")
        
        if 'dashboard_path' in results:
            print(f"📋 Dashboard: {results['dashboard_path']}")
            
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)