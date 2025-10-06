"""
Calibration Analysis for Adaptive Evaluation Suite
=================================================

Implements comprehensive calibration analysis including ECE, MCE,
temperature scaling, and reliability diagram generation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

from .metrics import TemperatureScaling, CalibrationMetrics

logger = logging.getLogger(__name__)


class CalibrationAnalyzer:
    """
    Comprehensive calibration analysis for model evaluation results.
    Handles ECE/MCE computation, temperature scaling, and reliability diagrams.
    """
    
    def __init__(self, results_dir: Optional[str] = None, num_bins: int = 15):
        self.results_dir = Path(results_dir) if results_dir else Path("./reports")
        self.num_bins = num_bins
        
    def extract_calibration_data(self, 
                               evaluation_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract calibration-related data from evaluation results.
        
        Args:
            evaluation_results: Results from evaluation pipeline
            
        Returns:
            Dict mapping model names to calibration data
        """
        calibration_data = {}
        
        for exp_id, result in evaluation_results.items():
            # Extract model name
            model_name = exp_id.replace('adaptive_eval_', '').replace('_', ' ').title()
            
            model_cal_data = {
                'in_domain_ece': None,
                'in_domain_mce': None,
                'validation_ece': None,
                'validation_mce': None,
                'temperature': 1.0,
                'ood_calibration': {}
            }
            
            # Extract in-domain calibration
            if ('aggregated_metrics' in result and 
                'in_domain' in result['aggregated_metrics']):
                in_domain = result['aggregated_metrics']['in_domain']
                model_cal_data['in_domain_ece'] = in_domain.get('ece', {}).get('mean')
                model_cal_data['in_domain_mce'] = in_domain.get('mce', {}).get('mean')
                
            # Extract validation calibration
            if ('aggregated_metrics' in result and 
                'validation' in result['aggregated_metrics']):
                validation = result['aggregated_metrics']['validation']
                model_cal_data['validation_ece'] = validation.get('ece', {}).get('mean')
                model_cal_data['validation_mce'] = validation.get('mce', {}).get('mean')
                
            # Extract temperature if available
            if 'temperature' in result:
                model_cal_data['temperature'] = result['temperature']
                
            # Extract OOD calibration data
            if ('aggregated_metrics' in result and 
                'ood_datasets' in result['aggregated_metrics']):
                ood_datasets = result['aggregated_metrics']['ood_datasets']
                
                for ood_name, ood_metrics in ood_datasets.items():
                    if isinstance(ood_metrics, dict):
                        model_cal_data['ood_calibration'][ood_name] = {
                            'ece': ood_metrics.get('ece', {}).get('mean'),
                            'mce': ood_metrics.get('mce', {}).get('mean')
                        }
                        
            calibration_data[model_name] = model_cal_data
            
        logger.info(f"Extracted calibration data for {len(calibration_data)} models")
        return calibration_data
        
    def analyze_calibration(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive calibration analysis.
        
        Args:
            evaluation_results: Results from evaluation pipeline
            
        Returns:
            Complete calibration analysis results
        """
        logger.info("Running calibration analysis...")
        
        # Extract calibration data
        calibration_data = self.extract_calibration_data(evaluation_results)
        
        if not calibration_data:
            return {'error': 'No calibration data extracted'}
            
        analysis_results = {
            'calibration_data': calibration_data,
            'calibration_summary': {},
            'temperature_scaling_effects': {},
            'ood_calibration_analysis': {},
            'calibration_rankings': {}
        }
        
        # 1. Calibration Summary Statistics
        analysis_results['calibration_summary'] = self._compute_calibration_summary(calibration_data)
        
        # 2. Temperature Scaling Effects Analysis
        analysis_results['temperature_scaling_effects'] = self._analyze_temperature_effects(calibration_data)
        
        # 3. OOD Calibration Analysis
        analysis_results['ood_calibration_analysis'] = self._analyze_ood_calibration(calibration_data)
        
        # 4. Calibration Rankings
        analysis_results['calibration_rankings'] = self._compute_calibration_rankings(calibration_data)
        
        # 5. Generate Plots
        plot_paths = self._generate_calibration_plots(analysis_results)
        analysis_results['plot_paths'] = plot_paths
        
        logger.info("Calibration analysis completed")
        return analysis_results
        
    def _compute_calibration_summary(self, 
                                   calibration_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics for calibration metrics."""
        
        summary = {
            'in_domain_statistics': {},
            'validation_statistics': {},
            'temperature_statistics': {},
            'overall_calibration_quality': {}
        }
        
        # Collect valid ECE/MCE values
        in_domain_ece = [data['in_domain_ece'] for data in calibration_data.values() 
                        if data['in_domain_ece'] is not None]
        in_domain_mce = [data['in_domain_mce'] for data in calibration_data.values() 
                        if data['in_domain_mce'] is not None]
        
        validation_ece = [data['validation_ece'] for data in calibration_data.values() 
                         if data['validation_ece'] is not None]
        validation_mce = [data['validation_mce'] for data in calibration_data.values() 
                         if data['validation_mce'] is not None]
        
        temperatures = [data['temperature'] for data in calibration_data.values() 
                       if data['temperature'] != 1.0]
        
        # In-domain statistics
        if in_domain_ece:
            summary['in_domain_statistics'] = {
                'ece': {
                    'mean': np.mean(in_domain_ece),
                    'std': np.std(in_domain_ece),
                    'min': np.min(in_domain_ece),
                    'max': np.max(in_domain_ece),
                    'count': len(in_domain_ece)
                },
                'mce': {
                    'mean': np.mean(in_domain_mce) if in_domain_mce else 0.0,
                    'std': np.std(in_domain_mce) if in_domain_mce else 0.0,
                    'min': np.min(in_domain_mce) if in_domain_mce else 0.0,
                    'max': np.max(in_domain_mce) if in_domain_mce else 0.0,
                    'count': len(in_domain_mce)
                }
            }
            
        # Validation statistics
        if validation_ece:
            summary['validation_statistics'] = {
                'ece': {
                    'mean': np.mean(validation_ece),
                    'std': np.std(validation_ece),
                    'min': np.min(validation_ece),
                    'max': np.max(validation_ece),
                    'count': len(validation_ece)
                },
                'mce': {
                    'mean': np.mean(validation_mce) if validation_mce else 0.0,
                    'std': np.std(validation_mce) if validation_mce else 0.0,
                    'min': np.min(validation_mce) if validation_mce else 0.0,
                    'max': np.max(validation_mce) if validation_mce else 0.0,
                    'count': len(validation_mce)
                }
            }
            
        # Temperature statistics
        if temperatures:
            summary['temperature_statistics'] = {
                'mean': np.mean(temperatures),
                'std': np.std(temperatures),
                'min': np.min(temperatures),
                'max': np.max(temperatures),
                'count': len(temperatures),
                'scaling_applied_ratio': len(temperatures) / len(calibration_data)
            }
        else:
            summary['temperature_statistics'] = {
                'mean': 1.0, 'std': 0.0, 'min': 1.0, 'max': 1.0, 
                'count': 0, 'scaling_applied_ratio': 0.0
            }
            
        # Overall calibration quality assessment
        if in_domain_ece:
            avg_ece = np.mean(in_domain_ece)
            if avg_ece <= 0.05:
                quality = 'Excellent'
            elif avg_ece <= 0.10:
                quality = 'Good'
            elif avg_ece <= 0.15:
                quality = 'Fair'
            else:
                quality = 'Poor'
                
            summary['overall_calibration_quality'] = {
                'average_ece': avg_ece,
                'quality_assessment': quality,
                'well_calibrated_models': sum(1 for ece in in_domain_ece if ece <= 0.10),
                'poorly_calibrated_models': sum(1 for ece in in_domain_ece if ece > 0.15)
            }
            
        return summary
        
    def _analyze_temperature_effects(self, 
                                   calibration_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the effects of temperature scaling on calibration."""
        
        effects = {
            'models_with_scaling': [],
            'temperature_improvements': [],
            'average_improvement': 0.0,
            'scaling_effectiveness': {}
        }
        
        for model_name, data in calibration_data.items():
            temperature = data['temperature']
            
            if temperature != 1.0:
                effects['models_with_scaling'].append({
                    'model': model_name,
                    'temperature': temperature,
                    'val_ece': data['validation_ece'],
                    'in_domain_ece': data['in_domain_ece']
                })
                
                # Calculate improvement if both ECE values available
                if (data['validation_ece'] is not None and 
                    data['in_domain_ece'] is not None):
                    # Assume validation ECE is before scaling, in_domain is after
                    improvement = data['validation_ece'] - data['in_domain_ece']
                    effects['temperature_improvements'].append(improvement)
                    
        # Calculate average improvement
        if effects['temperature_improvements']:
            effects['average_improvement'] = np.mean(effects['temperature_improvements'])
            
            # Scaling effectiveness categories
            significant_improvements = sum(1 for imp in effects['temperature_improvements'] if imp > 0.02)
            total_scaled = len(effects['temperature_improvements'])
            
            effects['scaling_effectiveness'] = {
                'models_with_significant_improvement': significant_improvements,
                'effectiveness_ratio': significant_improvements / max(total_scaled, 1),
                'total_models_scaled': total_scaled
            }
            
        return effects
        
    def _analyze_ood_calibration(self, 
                               calibration_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze calibration performance on OOD datasets."""
        
        ood_analysis = {
            'ood_degradation': {},
            'corruption_sensitivity': {},
            'calibration_robustness_ranking': []
        }
        
        # Analyze calibration degradation from in-domain to OOD
        for model_name, data in calibration_data.items():
            if data['in_domain_ece'] is None:
                continue
                
            in_domain_ece = data['in_domain_ece']
            ood_calibrations = data['ood_calibration']
            
            model_degradations = []
            
            for ood_name, ood_data in ood_calibrations.items():
                if ood_data['ece'] is not None:
                    degradation = ood_data['ece'] - in_domain_ece
                    model_degradations.append(degradation)
                    
            if model_degradations:
                ood_analysis['ood_degradation'][model_name] = {
                    'mean_degradation': np.mean(model_degradations),
                    'max_degradation': np.max(model_degradations),
                    'degradation_std': np.std(model_degradations),
                    'num_ood_evaluated': len(model_degradations)
                }
                
        # Compute calibration robustness ranking
        robustness_scores = []
        for model_name, degradation_data in ood_analysis['ood_degradation'].items():
            # Lower degradation = better robustness
            robustness_score = -degradation_data['mean_degradation']
            robustness_scores.append((model_name, robustness_score))
            
        robustness_scores.sort(key=lambda x: x[1], reverse=True)
        ood_analysis['calibration_robustness_ranking'] = robustness_scores
        
        # Analyze corruption sensitivity
        corruption_sensitivity = {}
        
        # Group OOD results by corruption type
        for model_name, data in calibration_data.items():
            for ood_name, ood_data in data['ood_calibration'].items():
                if ood_data['ece'] is not None:
                    # Extract corruption type from ood_name
                    corruption_type = ood_name.split('_severity_')[0]
                    
                    if corruption_type not in corruption_sensitivity:
                        corruption_sensitivity[corruption_type] = []
                        
                    corruption_sensitivity[corruption_type].append(ood_data['ece'])
                    
        # Compute statistics for each corruption type
        for corruption_type, ece_values in corruption_sensitivity.items():
            corruption_sensitivity[corruption_type] = {
                'mean_ece': np.mean(ece_values),
                'std_ece': np.std(ece_values),
                'worst_ece': np.max(ece_values),
                'best_ece': np.min(ece_values),
                'num_evaluations': len(ece_values)
            }
            
        ood_analysis['corruption_sensitivity'] = corruption_sensitivity
        
        return ood_analysis
        
    def _compute_calibration_rankings(self, 
                                    calibration_data: Dict[str, Dict[str, Any]]) -> Dict[str, List]:
        """Compute calibration rankings for different metrics."""
        
        rankings = {}
        
        # In-domain ECE ranking (lower is better)
        ece_ranking = []
        for model_name, data in calibration_data.items():
            if data['in_domain_ece'] is not None:
                ece_ranking.append((model_name, data['in_domain_ece']))
                
        ece_ranking.sort(key=lambda x: x[1])
        rankings['ece_ranking'] = ece_ranking
        
        # In-domain MCE ranking (lower is better)
        mce_ranking = []
        for model_name, data in calibration_data.items():
            if data['in_domain_mce'] is not None:
                mce_ranking.append((model_name, data['in_domain_mce']))
                
        mce_ranking.sort(key=lambda x: x[1])
        rankings['mce_ranking'] = mce_ranking
        
        # Temperature scaling requirement ranking (lower temperature closer to 1.0 is better)
        temp_ranking = []
        for model_name, data in calibration_data.items():
            temp_deviation = abs(data['temperature'] - 1.0)
            temp_ranking.append((model_name, temp_deviation))
            
        temp_ranking.sort(key=lambda x: x[1])
        rankings['temperature_ranking'] = temp_ranking
        
        return rankings
        
    def _generate_calibration_plots(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate calibration analysis plots."""
        
        plot_paths = {}
        
        # Ensure plots directory exists
        plots_dir = self.results_dir / "calibration_analysis"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. ECE/MCE Comparison Plot
            plot_path = self._plot_calibration_comparison(
                analysis_results['calibration_data'],
                plots_dir / "calibration_comparison.png"
            )
            plot_paths['calibration_comparison'] = str(plot_path)
            
            # 2. Temperature Scaling Effects Plot
            if analysis_results['temperature_scaling_effects']['models_with_scaling']:
                plot_path = self._plot_temperature_effects(
                    analysis_results['temperature_scaling_effects'],
                    plots_dir / "temperature_scaling_effects.png"
                )
                plot_paths['temperature_effects'] = str(plot_path)
                
            # 3. OOD Calibration Degradation Plot
            if analysis_results['ood_calibration_analysis']['ood_degradation']:
                plot_path = self._plot_ood_calibration(
                    analysis_results['ood_calibration_analysis'],
                    plots_dir / "ood_calibration_degradation.png"
                )
                plot_paths['ood_calibration'] = str(plot_path)
                
            # 4. Calibration Summary Heatmap
            plot_path = self._plot_calibration_heatmap(
                analysis_results['calibration_data'],
                plots_dir / "calibration_heatmap.png"
            )
            plot_paths['calibration_heatmap'] = str(plot_path)
            
        except Exception as e:
            logger.error(f"Calibration plot generation failed: {e}")
            plot_paths['error'] = str(e)
            
        return plot_paths
        
    def _plot_calibration_comparison(self, 
                                   calibration_data: Dict[str, Dict[str, Any]], 
                                   save_path: Path) -> Path:
        """Plot ECE vs MCE comparison for all models."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        models = []
        ece_values = []
        mce_values = []
        
        for model_name, data in calibration_data.items():
            if data['in_domain_ece'] is not None and data['in_domain_mce'] is not None:
                models.append(model_name)
                ece_values.append(data['in_domain_ece'])
                mce_values.append(data['in_domain_mce'])
                
        if models:
            # Scatter plot
            colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
            
            for i, (model, ece, mce) in enumerate(zip(models, ece_values, mce_values)):
                ax.scatter(ece, mce, c=[colors[i]], s=100, alpha=0.7, edgecolor='black')
                ax.annotate(model, (ece, mce), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
                           
            # Add diagonal line (ECE = MCE)
            max_val = max(max(ece_values), max(mce_values))
            ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='ECE = MCE')
            
            # Add calibration quality regions
            ax.axvline(x=0.05, color='green', linestyle=':', alpha=0.7, label='ECE ≤ 0.05 (Excellent)')
            ax.axvline(x=0.10, color='orange', linestyle=':', alpha=0.7, label='ECE ≤ 0.10 (Good)')
            ax.axvline(x=0.15, color='red', linestyle=':', alpha=0.7, label='ECE ≤ 0.15 (Fair)')
            
        ax.set_xlabel('Expected Calibration Error (ECE)', fontsize=12)
        ax.set_ylabel('Maximum Calibration Error (MCE)', fontsize=12)
        ax.set_title('Model Calibration Comparison (ECE vs MCE)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Calibration comparison plot saved to {save_path}")
        return save_path
        
    def _plot_temperature_effects(self, 
                                temperature_effects: Dict[str, Any], 
                                save_path: Path) -> Path:
        """Plot temperature scaling effects."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        models_data = temperature_effects['models_with_scaling']
        
        if models_data:
            # Plot 1: Temperature values
            models = [data['model'] for data in models_data]
            temperatures = [data['temperature'] for data in models_data]
            
            bars1 = ax1.bar(range(len(models)), temperatures, color='lightcoral', edgecolor='darkred')
            ax1.set_xticks(range(len(models)))
            ax1.set_xticklabels(models, rotation=45, ha='right')
            ax1.set_ylabel('Temperature Value')
            ax1.set_title('Temperature Scaling Values by Model', fontweight='bold')
            ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='T = 1.0 (No scaling)')
            ax1.grid(axis='y', alpha=0.3)
            ax1.legend()
            
            # Add value labels on bars
            for bar, temp in zip(bars1, temperatures):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{temp:.2f}', ha='center', va='bottom', fontsize=9)
                        
            # Plot 2: ECE improvements (if available)
            improvements = temperature_effects['temperature_improvements']
            if improvements:
                models_with_improvements = models[:len(improvements)]
                
                colors = ['green' if imp > 0 else 'red' for imp in improvements]
                bars2 = ax2.bar(range(len(models_with_improvements)), improvements, 
                              color=colors, alpha=0.7, edgecolor='black')
                
                ax2.set_xticks(range(len(models_with_improvements)))
                ax2.set_xticklabels(models_with_improvements, rotation=45, ha='right')
                ax2.set_ylabel('ECE Improvement')
                ax2.set_title('Temperature Scaling ECE Improvements', fontweight='bold')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
                ax2.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar, imp in zip(bars2, improvements):
                    height = bar.get_height()
                    y_pos = height + 0.001 if height >= 0 else height - 0.005
                    ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                            f'{imp:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                            fontsize=9)
            else:
                ax2.text(0.5, 0.5, 'No improvement data available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Temperature Scaling ECE Improvements', fontweight='bold')
                
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Temperature effects plot saved to {save_path}")
        return save_path
        
    def _plot_ood_calibration(self, 
                            ood_analysis: Dict[str, Any], 
                            save_path: Path) -> Path:
        """Plot OOD calibration degradation analysis."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Calibration degradation by model
        degradation_data = ood_analysis['ood_degradation']
        
        if degradation_data:
            models = list(degradation_data.keys())
            mean_degradations = [data['mean_degradation'] for data in degradation_data.values()]
            max_degradations = [data['max_degradation'] for data in degradation_data.values()]
            
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, mean_degradations, width, 
                           label='Mean Degradation', color='skyblue', edgecolor='navy')
            bars2 = ax1.bar(x + width/2, max_degradations, width,
                           label='Max Degradation', color='lightcoral', edgecolor='darkred')
            
            ax1.set_xlabel('Models')
            ax1.set_ylabel('ECE Degradation')
            ax1.set_title('OOD Calibration Degradation by Model', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.7)
            
        # Plot 2: Corruption sensitivity
        corruption_sensitivity = ood_analysis.get('corruption_sensitivity', {})
        
        if corruption_sensitivity:
            corruptions = list(corruption_sensitivity.keys())
            mean_eces = [data['mean_ece'] for data in corruption_sensitivity.values()]
            
            bars = ax2.bar(corruptions, mean_eces, color='orange', alpha=0.7, edgecolor='darkorange')
            ax2.set_xlabel('Corruption Types')
            ax2.set_ylabel('Mean ECE')
            ax2.set_title('Calibration Sensitivity by Corruption Type', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, ece in zip(bars, mean_eces):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{ece:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No corruption sensitivity data available',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Calibration Sensitivity by Corruption Type', fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"OOD calibration plot saved to {save_path}")
        return save_path
        
    def _plot_calibration_heatmap(self, 
                                calibration_data: Dict[str, Dict[str, Any]], 
                                save_path: Path) -> Path:
        """Plot calibration heatmap across models and metrics."""
        
        # Prepare data matrix
        models = list(calibration_data.keys())
        metrics = ['In-Domain ECE', 'In-Domain MCE', 'Temperature', 'Mean OOD ECE']
        
        data_matrix = []
        
        for model_name in models:
            data = calibration_data[model_name]
            
            row = [
                data['in_domain_ece'] if data['in_domain_ece'] is not None else 0.0,
                data['in_domain_mce'] if data['in_domain_mce'] is not None else 0.0,
                data['temperature'],
                np.mean([ood_data['ece'] for ood_data in data['ood_calibration'].values() 
                        if ood_data['ece'] is not None]) if data['ood_calibration'] else 0.0
            ]
            data_matrix.append(row)
            
        data_matrix = np.array(data_matrix)
        
        if data_matrix.size > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            im = ax.imshow(data_matrix.T, cmap='RdYlBu_r', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(models)))
            ax.set_yticks(np.arange(len(metrics)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_yticklabels(metrics)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Metric Value', fontsize=12)
            
            # Add text annotations
            for i in range(len(models)):
                for j in range(len(metrics)):
                    text = ax.text(i, j, f'{data_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=8)
                                 
            ax.set_title('Calibration Metrics Heatmap', fontsize=14, fontweight='bold')
            
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No calibration data available for heatmap',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Calibration Metrics Heatmap', fontsize=14, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Calibration heatmap saved to {save_path}")
        return save_path


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test calibration analyzer
    analyzer = CalibrationAnalyzer("./test_results")
    
    # Mock evaluation results
    mock_results = {
        'adaptive_eval_baseline': {
            'aggregated_metrics': {
                'in_domain': {
                    'ece': {'mean': 0.08},
                    'mce': {'mean': 0.15}
                },
                'validation': {
                    'ece': {'mean': 0.12},
                    'mce': {'mean': 0.20}
                },
                'ood_datasets': {
                    'gaussian_noise_severity_3': {
                        'ece': {'mean': 0.14},
                        'mce': {'mean': 0.25}
                    }
                }
            },
            'temperature': 1.2
        }
    }
    
    analysis = analyzer.analyze_calibration(mock_results)
    print(f"Calibration analysis completed:")
    print(f"- Models analyzed: {len(analysis.get('calibration_data', {}))}")
    print(f"- Temperature scaling models: {len(analysis.get('temperature_scaling_effects', {}).get('models_with_scaling', []))}")
    print(f"- Plot paths: {list(analysis.get('plot_paths', {}).keys())}")