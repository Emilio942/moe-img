"""
Cost Analysis and Pareto Optimization for Adaptive Evaluation
===========================================================

Implements Pareto-front analysis for accuracy vs. cost trade-offs,
hypervolume computation, and cost-efficiency recommendations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ParettoAnalyzer:
    """
    Pareto-front analysis for multi-objective optimization.
    Handles accuracy vs. time/energy/memory trade-offs.
    """
    
    def __init__(self):
        self.pareto_fronts = {}
        
    def find_pareto_front(self, 
                         points: List[Tuple[float, float]],
                         maximize_both: bool = False) -> List[int]:
        """
        Find Pareto-optimal points.
        
        Args:
            points: List of (objective1, objective2) tuples
            maximize_both: If True, maximize both objectives; else maximize first, minimize second
            
        Returns:
            List of indices of Pareto-optimal points
        """
        if not points:
            return []
            
        points_array = np.array(points)
        n_points = len(points)
        
        # Determine dominance relationship
        pareto_indices = []
        
        for i in range(n_points):
            is_pareto = True
            
            for j in range(n_points):
                if i == j:
                    continue
                    
                if maximize_both:
                    # Both objectives to maximize
                    if (points_array[j][0] >= points_array[i][0] and 
                        points_array[j][1] >= points_array[i][1] and
                        (points_array[j][0] > points_array[i][0] or 
                         points_array[j][1] > points_array[i][1])):
                        is_pareto = False
                        break
                else:
                    # Maximize first, minimize second (accuracy vs cost)
                    if (points_array[j][0] >= points_array[i][0] and 
                        points_array[j][1] <= points_array[i][1] and
                        (points_array[j][0] > points_array[i][0] or 
                         points_array[j][1] < points_array[i][1])):
                        is_pareto = False
                        break
                        
            if is_pareto:
                pareto_indices.append(i)
                
        return pareto_indices
        
    def compute_hypervolume(self, 
                          pareto_points: List[Tuple[float, float]],
                          reference_point: Tuple[float, float]) -> float:
        """
        Compute hypervolume indicator for Pareto front.
        
        Args:
            pareto_points: List of Pareto-optimal points
            reference_point: Reference point (typically worst case)
            
        Returns:
            Hypervolume value
        """
        if not pareto_points:
            return 0.0
            
        # Sort points by first objective (accuracy)
        sorted_points = sorted(pareto_points, key=lambda x: x[0])
        
        hypervolume = 0.0
        prev_x = reference_point[0]
        
        for x, y in sorted_points:
            # Add rectangle area
            width = x - prev_x
            height = y - reference_point[1]
            
            if width > 0 and height > 0:
                hypervolume += width * height
                
            prev_x = x
            
        return hypervolume
        
    def analyze_pareto_front(self,
                           data: Dict[str, Tuple[float, float]],
                           objective_names: Tuple[str, str],
                           maximize_both: bool = False) -> Dict[str, Any]:
        """
        Complete Pareto analysis for given data.
        
        Args:
            data: Dict mapping model names to (obj1, obj2) tuples
            objective_names: Names of the two objectives
            maximize_both: Whether to maximize both objectives
            
        Returns:
            Analysis results with Pareto front and statistics
        """
        if not data:
            return {'error': 'No data provided'}
            
        model_names = list(data.keys())
        points = list(data.values())
        
        # Find Pareto front
        pareto_indices = self.find_pareto_front(points, maximize_both)
        pareto_models = [model_names[i] for i in pareto_indices]
        pareto_points = [points[i] for i in pareto_indices]
        
        # Compute reference point for hypervolume
        all_obj1 = [p[0] for p in points]
        all_obj2 = [p[1] for p in points]
        
        if maximize_both:
            ref_point = (min(all_obj1), min(all_obj2))
        else:
            ref_point = (min(all_obj1), max(all_obj2))
            
        # Compute hypervolume
        hypervolume = self.compute_hypervolume(pareto_points, ref_point)
        
        # Find best single-objective solutions
        best_obj1_idx = np.argmax(all_obj1)
        best_obj2_idx = np.argmin(all_obj2) if not maximize_both else np.argmax(all_obj2)
        
        analysis = {
            'pareto_front': {
                'models': pareto_models,
                'points': pareto_points,
                'indices': pareto_indices
            },
            'hypervolume': hypervolume,
            'reference_point': ref_point,
            'best_single_objective': {
                objective_names[0]: {
                    'model': model_names[best_obj1_idx],
                    'value': all_obj1[best_obj1_idx]
                },
                objective_names[1]: {
                    'model': model_names[best_obj2_idx], 
                    'value': all_obj2[best_obj2_idx]
                }
            },
            'statistics': {
                'total_models': len(model_names),
                'pareto_optimal_models': len(pareto_models),
                'pareto_efficiency_ratio': len(pareto_models) / len(model_names),
                f'{objective_names[0]}_range': (min(all_obj1), max(all_obj1)),
                f'{objective_names[1]}_range': (min(all_obj2), max(all_obj2))
            }
        }
        
        return analysis


class CostAnalyzer:
    """
    Comprehensive cost analysis for model evaluation results.
    Generates Pareto fronts, cost-efficiency metrics, and recommendations.
    """
    
    def __init__(self, results_dir: Optional[str] = None):
        self.results_dir = Path(results_dir) if results_dir else Path("./reports")
        self.pareto_analyzer = ParettoAnalyzer()
        
    def extract_cost_metrics(self, evaluation_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Extract cost and accuracy metrics from evaluation results.
        
        Args:
            evaluation_results: Results from evaluation pipeline
            
        Returns:
            Dict mapping model names to cost/accuracy metrics
        """
        extracted_metrics = {}
        
        for exp_id, result in evaluation_results.items():
            if 'aggregated_metrics' not in result or 'in_domain' not in result['aggregated_metrics']:
                logger.warning(f"Missing metrics for experiment {exp_id}")
                continue
                
            in_domain = result['aggregated_metrics']['in_domain']
            
            # Extract core metrics with safety checks
            metrics = {
                'accuracy': in_domain.get('top1_accuracy', {}).get('mean', 0.0),
                'time_per_batch': in_domain.get('time_per_batch_median', {}).get('mean', float('inf')),
                'memory_peak_mb': in_domain.get('memory_peak_mb', {}).get('mean', float('inf')),
                'energy_proxy': in_domain.get('energy_proxy', {}).get('mean', float('inf')),
                'param_count': result.get('param_count', 0),
                'model_size_mb': result.get('model_size_mb', 0.0)
            }
            
            # Clean model name from experiment ID
            model_name = exp_id.replace('adaptive_eval_', '').replace('_', ' ').title()
            extracted_metrics[model_name] = metrics
            
        logger.info(f"Extracted metrics for {len(extracted_metrics)} models")
        return extracted_metrics
        
    def analyze_cost_efficiency(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive cost-efficiency analysis with Pareto fronts.
        
        Args:
            evaluation_results: Results from evaluation pipeline
            
        Returns:
            Complete cost analysis results
        """
        logger.info("Running cost-efficiency analysis...")
        
        # Extract metrics
        metrics = self.extract_cost_metrics(evaluation_results)
        
        if not metrics:
            return {'error': 'No valid metrics extracted'}
            
        analysis_results = {
            'extracted_metrics': metrics,
            'pareto_analysis': {},
            'efficiency_rankings': {},
            'cost_summaries': {}
        }
        
        # 1. Accuracy vs Time Pareto Analysis
        acc_time_data = {}
        for model, model_metrics in metrics.items():
            if model_metrics['time_per_batch'] != float('inf'):
                acc_time_data[model] = (
                    model_metrics['accuracy'],
                    model_metrics['time_per_batch']
                )
                
        if acc_time_data:
            acc_time_pareto = self.pareto_analyzer.analyze_pareto_front(
                acc_time_data,
                objective_names=('Accuracy', 'Time per Batch'),
                maximize_both=False  # Maximize accuracy, minimize time
            )
            analysis_results['pareto_analysis']['accuracy_vs_time'] = acc_time_pareto
            
        # 2. Accuracy vs Energy Pareto Analysis
        acc_energy_data = {}
        for model, model_metrics in metrics.items():
            if model_metrics['energy_proxy'] != float('inf'):
                acc_energy_data[model] = (
                    model_metrics['accuracy'],
                    model_metrics['energy_proxy']
                )
                
        if acc_energy_data:
            acc_energy_pareto = self.pareto_analyzer.analyze_pareto_front(
                acc_energy_data,
                objective_names=('Accuracy', 'Energy Proxy'),
                maximize_both=False
            )
            analysis_results['pareto_analysis']['accuracy_vs_energy'] = acc_energy_pareto
            
        # 3. Accuracy vs Memory Pareto Analysis  
        acc_memory_data = {}
        for model, model_metrics in metrics.items():
            if model_metrics['memory_peak_mb'] != float('inf'):
                acc_memory_data[model] = (
                    model_metrics['accuracy'],
                    model_metrics['memory_peak_mb']
                )
                
        if acc_memory_data:
            acc_memory_pareto = self.pareto_analyzer.analyze_pareto_front(
                acc_memory_data,
                objective_names=('Accuracy', 'Memory Peak MB'),
                maximize_both=False
            )
            analysis_results['pareto_analysis']['accuracy_vs_memory'] = acc_memory_pareto
            
        # 4. Efficiency Rankings
        analysis_results['efficiency_rankings'] = self._compute_efficiency_rankings(metrics)
        
        # 5. Cost Summaries
        analysis_results['cost_summaries'] = self._compute_cost_summaries(metrics)
        
        # 6. Generate Plots
        plot_paths = self._generate_cost_plots(analysis_results)
        analysis_results['plot_paths'] = plot_paths
        
        logger.info("Cost-efficiency analysis completed")
        return analysis_results
        
    def _compute_efficiency_rankings(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, List]:
        """Compute efficiency rankings for different cost metrics."""
        
        rankings = {}
        
        # Accuracy per unit time (higher is better)
        time_efficiency = []
        for model, model_metrics in metrics.items():
            if model_metrics['time_per_batch'] > 0:
                efficiency = model_metrics['accuracy'] / model_metrics['time_per_batch']
                time_efficiency.append((model, efficiency))
                
        time_efficiency.sort(key=lambda x: x[1], reverse=True)
        rankings['accuracy_per_time'] = time_efficiency
        
        # Accuracy per unit energy (higher is better)
        energy_efficiency = []
        for model, model_metrics in metrics.items():
            if model_metrics['energy_proxy'] > 0:
                efficiency = model_metrics['accuracy'] / model_metrics['energy_proxy']
                energy_efficiency.append((model, efficiency))
                
        energy_efficiency.sort(key=lambda x: x[1], reverse=True)
        rankings['accuracy_per_energy'] = energy_efficiency
        
        # Accuracy per MB memory (higher is better)
        memory_efficiency = []
        for model, model_metrics in metrics.items():
            if model_metrics['memory_peak_mb'] > 0:
                efficiency = model_metrics['accuracy'] / model_metrics['memory_peak_mb']
                memory_efficiency.append((model, efficiency))
                
        memory_efficiency.sort(key=lambda x: x[1], reverse=True)
        rankings['accuracy_per_memory'] = memory_efficiency
        
        return rankings
        
    def _compute_cost_summaries(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compute summary statistics for cost metrics."""
        
        summaries = {}
        
        cost_metrics = ['time_per_batch', 'energy_proxy', 'memory_peak_mb', 'model_size_mb']
        
        for cost_metric in cost_metrics:
            values = []
            for model_metrics in metrics.values():
                value = model_metrics.get(cost_metric, 0.0)
                if value != float('inf') and value > 0:
                    values.append(value)
                    
            if values:
                summaries[cost_metric] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'range_ratio': max(values) / min(values) if min(values) > 0 else float('inf')
                }
            else:
                summaries[cost_metric] = {
                    'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'range_ratio': 1.0
                }
                
        return summaries
        
    def _generate_cost_plots(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate cost analysis plots."""
        
        plot_paths = {}
        
        # Ensure results directory exists
        plots_dir = self.results_dir / "cost_analysis"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Accuracy vs Time Pareto Plot
            if 'accuracy_vs_time' in analysis_results['pareto_analysis']:
                pareto_data = analysis_results['pareto_analysis']['accuracy_vs_time']
                plot_path = self._plot_pareto_front(
                    pareto_data, 
                    'Accuracy (%)', 'Time per Batch (s)',
                    'Accuracy vs Time Trade-off',
                    plots_dir / "pareto_acc_time.png"
                )
                plot_paths['pareto_acc_time'] = str(plot_path)
                
            # 2. Accuracy vs Energy Pareto Plot
            if 'accuracy_vs_energy' in analysis_results['pareto_analysis']:
                pareto_data = analysis_results['pareto_analysis']['accuracy_vs_energy']
                plot_path = self._plot_pareto_front(
                    pareto_data,
                    'Accuracy (%)', 'Energy Proxy (units)',
                    'Accuracy vs Energy Trade-off',
                    plots_dir / "pareto_acc_energy.png"
                )
                plot_paths['pareto_acc_energy'] = str(plot_path)
                
            # 3. Efficiency Rankings Plot
            if 'efficiency_rankings' in analysis_results:
                plot_path = self._plot_efficiency_rankings(
                    analysis_results['efficiency_rankings'],
                    plots_dir / "efficiency_rankings.png"
                )
                plot_paths['efficiency_rankings'] = str(plot_path)
                
            # 4. Hypervolume Evolution Plot
            plot_path = self._plot_hypervolume_comparison(
                analysis_results['pareto_analysis'],
                plots_dir / "hypervolume_comparison.png"
            )
            plot_paths['hypervolume_comparison'] = str(plot_path)
            
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
            plot_paths['error'] = str(e)
            
        return plot_paths
        
    def _plot_pareto_front(self, 
                          pareto_data: Dict[str, Any],
                          xlabel: str, ylabel: str, title: str,
                          save_path: Path) -> Path:
        """Plot Pareto front analysis."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract data
        pareto_models = pareto_data['pareto_front']['models']
        pareto_points = pareto_data['pareto_front']['points']
        
        # Plot all points
        all_models = list(pareto_data.get('all_data', {}).keys())
        all_points = list(pareto_data.get('all_data', {}).values())
        
        if not all_points:
            # Reconstruct from pareto points if all_data not available
            all_models = pareto_models
            all_points = pareto_points
            
        # Scatter plot
        for i, (model, point) in enumerate(zip(all_models, all_points)):
            color = 'red' if model in pareto_models else 'lightblue'
            size = 100 if model in pareto_models else 50
            ax.scatter(point[0], point[1], c=color, s=size, alpha=0.7, edgecolor='black')
            
            # Label Pareto-optimal points
            if model in pareto_models:
                ax.annotate(model, (point[0], point[1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
                
        # Draw Pareto front line
        if len(pareto_points) > 1:
            sorted_pareto = sorted(pareto_points, key=lambda x: x[0])
            pareto_x = [p[0] for p in sorted_pareto]
            pareto_y = [p[1] for p in sorted_pareto]
            ax.plot(pareto_x, pareto_y, 'r--', alpha=0.7, linewidth=2, label='Pareto Front')
            
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add hypervolume annotation
        if 'hypervolume' in pareto_data:
            ax.text(0.02, 0.98, f"Hypervolume: {pareto_data['hypervolume']:.3f}",
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                   
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Pareto plot saved to {save_path}")
        return save_path
        
    def _plot_efficiency_rankings(self, rankings: Dict[str, List], save_path: Path) -> Path:
        """Plot efficiency rankings comparison."""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        ranking_types = ['accuracy_per_time', 'accuracy_per_energy', 'accuracy_per_memory']
        titles = ['Accuracy per Time', 'Accuracy per Energy', 'Accuracy per Memory']
        
        for i, (ranking_type, title) in enumerate(zip(ranking_types, titles)):
            if ranking_type in rankings:
                ranking_data = rankings[ranking_type]
                
                models = [item[0] for item in ranking_data[:5]]  # Top 5
                scores = [item[1] for item in ranking_data[:5]]
                
                axes[i].barh(range(len(models)), scores, color='skyblue', edgecolor='navy')
                axes[i].set_yticks(range(len(models)))
                axes[i].set_yticklabels(models, fontsize=8)
                axes[i].set_xlabel('Efficiency Score')
                axes[i].set_title(title, fontweight='bold')
                axes[i].grid(axis='x', alpha=0.3)
                
                # Add score labels
                for j, score in enumerate(scores):
                    axes[i].text(score + 0.001, j, f'{score:.3f}', 
                               va='center', fontsize=8)
                               
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Efficiency rankings plot saved to {save_path}")
        return save_path
        
    def _plot_hypervolume_comparison(self, pareto_analyses: Dict[str, Any], save_path: Path) -> Path:
        """Plot hypervolume comparison across different trade-offs."""
        
        hypervolumes = {}
        for analysis_name, analysis_data in pareto_analyses.items():
            if 'hypervolume' in analysis_data:
                clean_name = analysis_name.replace('accuracy_vs_', '').replace('_', ' ').title()
                hypervolumes[clean_name] = analysis_data['hypervolume']
                
        if not hypervolumes:
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No hypervolume data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Hypervolume Comparison', fontweight='bold')
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            names = list(hypervolumes.keys())
            values = list(hypervolumes.values())
            
            bars = ax.bar(names, values, color='lightcoral', edgecolor='darkred')
            ax.set_ylabel('Hypervolume', fontsize=12)
            ax.set_title('Hypervolume Comparison Across Trade-offs', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
                       
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Hypervolume comparison plot saved to {save_path}")
        return save_path


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test Pareto analyzer
    pareto_analyzer = ParettoAnalyzer()
    
    # Test data: (accuracy, time)
    test_data = {
        'ModelA': (85.0, 0.05),
        'ModelB': (87.0, 0.08), 
        'ModelC': (90.0, 0.12),
        'ModelD': (88.0, 0.06),
        'ModelE': (82.0, 0.03)
    }
    
    analysis = pareto_analyzer.analyze_pareto_front(
        test_data, 
        ('Accuracy', 'Time'),
        maximize_both=False
    )
    
    print("Pareto Analysis Results:")
    print(f"Pareto models: {analysis['pareto_front']['models']}")
    print(f"Hypervolume: {analysis['hypervolume']:.4f}")
    print(f"Pareto efficiency ratio: {analysis['statistics']['pareto_efficiency_ratio']:.2f}")
    
    # Test cost analyzer
    cost_analyzer = CostAnalyzer("./test_results")
    
    # Mock evaluation results
    mock_results = {
        'adaptive_eval_baseline': {
            'aggregated_metrics': {
                'in_domain': {
                    'top1_accuracy': {'mean': 85.0},
                    'time_per_batch_median': {'mean': 0.05},
                    'memory_peak_mb': {'mean': 512.0},
                    'energy_proxy': {'mean': 0.1}
                }
            },
            'param_count': 1000000,
            'model_size_mb': 4.0
        }
    }
    
    cost_analysis = cost_analyzer.analyze_cost_efficiency(mock_results)
    print(f"\nCost analysis completed with {len(cost_analysis.get('extracted_metrics', {}))} models")