"""
Report Generation for Adaptive Evaluation Suite
==============================================

Comprehensive report generation including markdown reports,
dashboard creation, and result visualization.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Comprehensive report generator for adaptive evaluation results.
    Creates markdown reports, dashboards, and summary visualizations.
    """
    
    def __init__(self, results_root: str = "./reports"):
        self.results_root = Path(results_root)
        self.results_root.mkdir(parents=True, exist_ok=True)
        
    def generate_comprehensive_dashboard(self, 
                                       final_report: Dict[str, Any]) -> Path:
        """
        Generate comprehensive markdown dashboard with all results.
        
        Args:
            final_report: Complete evaluation results
            
        Returns:
            Path to generated dashboard
        """
        logger.info("Generating comprehensive adaptive evaluation dashboard...")
        
        dashboard_path = self.results_root / "adaptive_dashboard.md"
        
        with open(dashboard_path, 'w') as f:
            self._write_dashboard_header(f, final_report)
            self._write_execution_summary(f, final_report)
            self._write_model_comparison_table(f, final_report)
            self._write_cost_analysis_section(f, final_report)
            self._write_robustness_analysis_section(f, final_report)
            self._write_calibration_analysis_section(f, final_report)
            self._write_recommendations_section(f, final_report)
            self._write_detailed_results_section(f, final_report)
            self._write_phase8_completion_status(f, final_report)
            
        logger.info(f"Comprehensive dashboard generated at {dashboard_path}")
        return dashboard_path
        
    def _write_dashboard_header(self, f, final_report: Dict[str, Any]):
        """Write dashboard header and overview."""
        
        summary = final_report.get('evaluation_summary', {})
        
        f.write("# Phase 8: Adaptive Evaluation - Comprehensive Dashboard\n\n")
        f.write("## 🎯 Executive Summary\n\n")
        
        f.write(f"**Evaluation Period**: {summary.get('start_time', 'N/A')} - {summary.get('end_time', 'N/A')}\n\n")
        f.write(f"**Total Duration**: {summary.get('total_duration_seconds', 0):.0f} seconds\n\n")
        f.write(f"**Experiments Completed**: {summary.get('successful_experiments', 0)}/{summary.get('total_experiments', 0)}\n\n")
        f.write(f"**Seeds Evaluated**: {summary.get('seeds_evaluated', [])}\n\n")
        f.write(f"**Quick Mode**: {'Yes' if summary.get('quick_mode', False) else 'No'}\n\n")
        
        if summary.get('failed_experiments'):
            f.write(f"**⚠️ Failed Experiments**: {', '.join(summary['failed_experiments'])}\n\n")
            
        f.write("---\n\n")
        
    def _write_execution_summary(self, f, final_report: Dict[str, Any]):
        """Write execution summary section."""
        
        f.write("## 📊 Execution Summary\n\n")
        
        batch_results = final_report.get('batch_results', {})
        results = batch_results.get('results', {})
        
        # Count model types
        model_types = {}
        for exp_id in results.keys():
            model_type = exp_id.replace('adaptive_eval_', '').replace('_', ' ').title()
            model_types[model_type] = model_types.get(model_type, 0) + 1
            
        f.write("### Model Variants Evaluated\n\n")
        for model_type, count in model_types.items():
            f.write(f"- **{model_type}**: {count} configuration(s)\n")
            
        f.write("\n")
        
        # Success rate summary
        total_exp = len(results)
        successful_exp = sum(1 for r in results.values() if r.get('status') != 'failed')
        success_rate = (successful_exp / max(total_exp, 1)) * 100
        
        f.write(f"### Success Metrics\n\n")
        f.write(f"- **Overall Success Rate**: {success_rate:.1f}%\n")
        f.write(f"- **Successful Evaluations**: {successful_exp}\n")
        f.write(f"- **Failed Evaluations**: {total_exp - successful_exp}\n\n")
        
        f.write("---\n\n")
        
    def _write_model_comparison_table(self, f, final_report: Dict[str, Any]):
        """Write comprehensive model comparison table."""
        
        f.write("## 🏆 Model Performance Comparison\n\n")
        
        batch_results = final_report.get('batch_results', {})
        results = batch_results.get('results', {})
        
        # Extract metrics for comparison table
        comparison_data = []
        
        for exp_id, result in results.items():
            if result.get('status') == 'failed':
                continue
                
            model_name = exp_id.replace('adaptive_eval_', '').replace('_', ' ').title()
            
            # Extract aggregated metrics
            agg_metrics = result.get('aggregated_metrics', {})
            in_domain = agg_metrics.get('in_domain', {})
            
            row = {
                'Model': model_name,
                'Accuracy (%)': self._extract_mean_value(in_domain.get('top1_accuracy', {}), 'N/A'),
                'Time/Batch (s)': self._extract_mean_value(in_domain.get('time_per_batch_median', {}), 'N/A'),
                'Memory (MB)': self._extract_mean_value(in_domain.get('memory_peak_mb', {}), 'N/A'),
                'Energy Proxy': self._extract_mean_value(in_domain.get('energy_proxy', {}), 'N/A'),
                'ECE': self._extract_mean_value(in_domain.get('ece', {}), 'N/A'),
                'Params (M)': f"{result.get('param_count', 0) / 1e6:.1f}" if result.get('param_count') else 'N/A',
                'Size (MB)': f"{result.get('model_size_mb', 0):.1f}" if result.get('model_size_mb') else 'N/A'
            }
            
            # Add robustness if available
            if 'robustness_summary' in agg_metrics:
                rob_summary = agg_metrics['robustness_summary']
                row['Robustness'] = self._extract_mean_value(rob_summary.get('relative_robustness_mean', {}), 'N/A')
                
            comparison_data.append(row)
            
        if comparison_data:
            # Create markdown table
            headers = list(comparison_data[0].keys())
            
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(['---'] * len(headers)) + " |\n")
            
            for row in comparison_data:
                values = []
                for header in headers:
                    value = row[header]
                    if isinstance(value, float):
                        values.append(f"{value:.3f}")
                    else:
                        values.append(str(value))
                f.write("| " + " | ".join(values) + " |\n")
                
            f.write("\n")
            
            # Add table legend
            f.write("**Legend:**\n")
            f.write("- **Accuracy**: Top-1 accuracy on in-domain test set\n")
            f.write("- **Time/Batch**: Median time per batch (excluding warmup)\n")
            f.write("- **Memory**: Peak memory usage during evaluation\n")
            f.write("- **Energy Proxy**: Simplified energy consumption estimate\n")
            f.write("- **ECE**: Expected Calibration Error (lower is better)\n")
            f.write("- **Robustness**: Relative robustness score (OOD/ID accuracy ratio)\n\n")
        else:
            f.write("*No comparison data available*\n\n")
            
        f.write("---\n\n")
        
    def _write_cost_analysis_section(self, f, final_report: Dict[str, Any]):
        """Write cost analysis section."""
        
        f.write("## 💰 Cost-Efficiency Analysis\n\n")
        
        analysis_results = final_report.get('analysis_results', {})
        cost_analysis = analysis_results.get('cost_efficiency', {})
        
        if 'error' in cost_analysis:
            f.write(f"*Cost analysis failed: {cost_analysis['error']}*\n\n")
            f.write("---\n\n")
            return
            
        # Pareto analysis summary
        pareto_analysis = cost_analysis.get('pareto_analysis', {})
        
        f.write("### 📈 Pareto-Optimal Analysis\n\n")
        
        for trade_off, analysis_data in pareto_analysis.items():
            if 'pareto_front' in analysis_data:
                clean_name = trade_off.replace('accuracy_vs_', '').replace('_', ' ').title()
                f.write(f"#### {clean_name} Trade-off\n\n")
                
                pareto_models = analysis_data['pareto_front']['models']
                hypervolume = analysis_data.get('hypervolume', 0.0)
                
                f.write(f"- **Pareto-Optimal Models**: {', '.join(pareto_models)}\n")
                f.write(f"- **Hypervolume**: {hypervolume:.4f}\n")
                
                # Best single objectives
                if 'best_single_objective' in analysis_data:
                    best_obj = analysis_data['best_single_objective']
                    for obj_name, obj_data in best_obj.items():
                        f.write(f"- **Best {obj_name}**: {obj_data['model']} ({obj_data['value']:.3f})\n")
                        
                f.write("\n")
                
        # Efficiency rankings
        efficiency_rankings = cost_analysis.get('efficiency_rankings', {})
        
        f.write("### 🏃 Efficiency Rankings\n\n")
        
        for ranking_name, ranking_data in efficiency_rankings.items():
            clean_name = ranking_name.replace('accuracy_per_', '').replace('_', ' ').title()
            f.write(f"#### {clean_name} Efficiency\n\n")
            
            for i, (model, score) in enumerate(ranking_data[:3], 1):  # Top 3
                f.write(f"{i}. **{model}**: {score:.4f}\n")
                
            f.write("\n")
            
        # Plot references
        plot_paths = cost_analysis.get('plot_paths', {})
        if plot_paths:
            f.write("### 📊 Generated Plots\n\n")
            for plot_name, plot_path in plot_paths.items():
                if 'error' not in plot_name:
                    clean_name = plot_name.replace('_', ' ').title()
                    f.write(f"- **{clean_name}**: `{plot_path}`\n")
            f.write("\n")
            
        f.write("---\n\n")
        
    def _write_robustness_analysis_section(self, f, final_report: Dict[str, Any]):
        """Write robustness analysis section."""
        
        f.write("## 🛡️ Robustness Analysis\n\n")
        
        analysis_results = final_report.get('analysis_results', {})
        robustness_analysis = analysis_results.get('robustness', {})
        
        if not robustness_analysis or 'error' in robustness_analysis:
            f.write("*No robustness analysis data available*\n\n")
            f.write("---\n\n")
            return
            
        # Overall robustness summary
        summary = robustness_analysis.get('summary', {})
        
        f.write("### 📊 Overall Robustness Statistics\n\n")
        
        if summary:
            f.write(f"- **Best Relative Robustness**: {summary.get('best_relative_robustness', 0.0):.3f}\n")
            f.write(f"- **Worst Relative Robustness**: {summary.get('worst_relative_robustness', 0.0):.3f}\n")
            f.write(f"- **Mean Accuracy Drop**: {summary.get('mean_accuracy_drop', 0.0):.1f}%\n")
            f.write(f"- **Robustness Range**: {summary.get('robustness_range', 0.0):.3f}\n\n")
            
        # Per-model robustness
        by_model = robustness_analysis.get('by_model', {})
        
        if by_model:
            f.write("### 🔍 Per-Model Robustness\n\n")
            
            # Sort by relative robustness
            sorted_models = sorted(
                by_model.items(),
                key=lambda x: x[1].get('relative_robustness_mean', 0.0),
                reverse=True
            )
            
            f.write("| Model | Relative Robustness | Accuracy Drop (%) | Confidence Robustness |\n")
            f.write("| --- | --- | --- | --- |\n")
            
            for model_name, rob_data in sorted_models:
                rel_rob = rob_data.get('relative_robustness_mean', 0.0)
                acc_drop = rob_data.get('accuracy_drop_mean', 0.0)
                conf_rob = rob_data.get('confidence_robustness', 0.0)
                
                f.write(f"| {model_name} | {rel_rob:.3f} | {acc_drop:.1f} | {conf_rob:.3f} |\n")
                
            f.write("\n")
            
        f.write("**Note**: Relative Robustness = OOD Accuracy / In-Domain Accuracy (higher is better)\n\n")
        f.write("---\n\n")
        
    def _write_calibration_analysis_section(self, f, final_report: Dict[str, Any]):
        """Write calibration analysis section."""
        
        f.write("## 🎯 Calibration Analysis\n\n")
        
        analysis_results = final_report.get('analysis_results', {})
        calibration_analysis = analysis_results.get('calibration', {})
        
        if not calibration_analysis or 'error' in calibration_analysis:
            f.write("*No calibration analysis data available*\n\n")
            f.write("---\n\n")
            return
            
        # Calibration summary
        cal_summary = calibration_analysis.get('calibration_summary', {})
        
        f.write("### 📊 Calibration Statistics\n\n")
        
        if 'overall_calibration_quality' in cal_summary:
            quality = cal_summary['overall_calibration_quality']
            f.write(f"- **Overall Calibration Quality**: {quality.get('quality_assessment', 'Unknown')}\n")
            f.write(f"- **Average ECE**: {quality.get('average_ece', 0.0):.4f}\n")
            f.write(f"- **Well-Calibrated Models**: {quality.get('well_calibrated_models', 0)}\n")
            f.write(f"- **Poorly Calibrated Models**: {quality.get('poorly_calibrated_models', 0)}\n\n")
            
        # Temperature scaling effects
        temp_effects = calibration_analysis.get('temperature_scaling_effects', {})
        
        if temp_effects.get('models_with_scaling'):
            f.write("### 🌡️ Temperature Scaling Effects\n\n")
            f.write(f"- **Models with Temperature Scaling**: {len(temp_effects['models_with_scaling'])}\n")
            f.write(f"- **Average ECE Improvement**: {temp_effects.get('average_improvement', 0.0):.4f}\n")
            
            effectiveness = temp_effects.get('scaling_effectiveness', {})
            if effectiveness:
                f.write(f"- **Significant Improvements**: {effectiveness.get('models_with_significant_improvement', 0)}\n")
                f.write(f"- **Effectiveness Ratio**: {effectiveness.get('effectiveness_ratio', 0.0):.1%}\n")
                
            f.write("\n")
            
        # Calibration rankings
        rankings = calibration_analysis.get('calibration_rankings', {})
        
        if rankings:
            f.write("### 🏆 Calibration Rankings\n\n")
            
            # ECE ranking (top 3)
            if 'ece_ranking' in rankings:
                f.write("#### Best Calibrated Models (ECE)\n\n")
                for i, (model, ece) in enumerate(rankings['ece_ranking'][:3], 1):
                    f.write(f"{i}. **{model}**: ECE = {ece:.4f}\n")
                f.write("\n")
                
        # Plot references
        plot_paths = calibration_analysis.get('plot_paths', {})
        if plot_paths:
            f.write("### 📊 Generated Plots\n\n")
            for plot_name, plot_path in plot_paths.items():
                if 'error' not in plot_name:
                    clean_name = plot_name.replace('_', ' ').title()
                    f.write(f"- **{clean_name}**: `{plot_path}`\n")
            f.write("\n")
            
        f.write("---\n\n")
        
    def _write_recommendations_section(self, f, final_report: Dict[str, Any]):
        """Write recommendations section."""
        
        f.write("## 💡 Recommendations\n\n")
        
        recommendations = final_report.get('recommendations', {})
        
        if not recommendations:
            f.write("*No recommendations generated*\n\n")
            f.write("---\n\n")
            return
            
        # Three operation modes
        modes = ['quality_max_mode', 'balanced_mode', 'cost_saving_mode']
        mode_names = ['Quality-Focused', 'Balanced', 'Cost-Efficient']
        
        f.write("### 🎛️ Operational Recommendations\n\n")
        
        for mode, mode_name in zip(modes, mode_names):
            mode_data = recommendations.get(mode, {})
            
            if mode_data:
                f.write(f"#### {mode_name} Mode\n\n")
                f.write(f"- **Recommended Model**: {mode_data.get('recommended_model', 'N/A')}\n")
                
                if 'accuracy_score' in mode_data:
                    f.write(f"- **Accuracy Score**: {mode_data['accuracy_score']:.2f}%\n")
                if 'efficiency_score' in mode_data:
                    f.write(f"- **Efficiency Score**: {mode_data['efficiency_score']:.4f}\n")
                if 'balanced_score' in mode_data:
                    f.write(f"- **Balanced Score**: {mode_data['balanced_score']:.4f}\n")
                    
                f.write(f"- **Rationale**: {mode_data.get('rationale', 'N/A')}\n\n")
                
        # General insights
        insights = recommendations.get('general_insights', [])
        
        if insights:
            f.write("### 🔍 General Insights\n\n")
            for insight in insights:
                f.write(f"- {insight}\n")
            f.write("\n")
            
        f.write("---\n\n")
        
    def _write_detailed_results_section(self, f, final_report: Dict[str, Any]):
        """Write detailed results section."""
        
        f.write("## 📋 Detailed Results\n\n")
        
        batch_results = final_report.get('batch_results', {})
        
        f.write("### 📁 Generated Files\n\n")
        f.write("The following files have been generated during this evaluation:\n\n")
        
        # List key result files
        result_files = [
            "adaptive_evaluation_final_report.json",
            "cost_analysis/pareto_acc_time.png",
            "cost_analysis/pareto_acc_energy.png", 
            "cost_analysis/efficiency_rankings.png",
            "calibration_analysis/calibration_comparison.png",
            "calibration_analysis/calibration_heatmap.png"
        ]
        
        for file_name in result_files:
            f.write(f"- `{self.results_root}/{file_name}`\n")
            
        f.write("\n")
        
        f.write("### 🔢 Raw Data Access\n\n")
        f.write("For programmatic access to results:\n\n")
        f.write("```python\n")
        f.write("import json\n")
        f.write(f"with open('{self.results_root}/adaptive_evaluation_final_report.json') as f:\n")
        f.write("    results = json.load(f)\n")
        f.write("```\n\n")
        
        f.write("---\n\n")
        
    def _write_phase8_completion_status(self, f, final_report: Dict[str, Any]):
        """Write Phase 8 completion status."""
        
        f.write("## ✅ Phase 8 Completion Status\n\n")
        
        completion_status = final_report.get('phase8_completion_status', {})
        
        f.write("### 📊 Evaluation Components\n\n")
        
        components = [
            ('in_domain_evaluation', 'In-Domain Evaluation'),
            ('ood_evaluation', 'Out-of-Domain (OOD) Evaluation'),
            ('cost_analysis', 'Cost-Efficiency Analysis'),
            ('calibration_analysis', 'Calibration Analysis'),
            ('robustness_analysis', 'Robustness Analysis'),
            ('pareto_analysis', 'Pareto-Optimal Analysis'),
            ('export_validation', 'Model Export Validation'),
            ('comprehensive_dashboard', 'Comprehensive Dashboard')
        ]
        
        for component_key, component_name in components:
            status = completion_status.get(component_key, False)
            icon = "✅" if status else "❌"
            f.write(f"- {icon} **{component_name}**: {'Complete' if status else 'Incomplete'}\n")
            
        f.write("\n")
        
        # Overall completion
        completed_components = sum(1 for _, component_key in components if completion_status.get(component_key, False))
        total_components = len(components)
        completion_rate = (completed_components / total_components) * 100
        
        f.write(f"### 🎯 Overall Completion: {completion_rate:.1f}% ({completed_components}/{total_components})\n\n")
        
        if completion_rate >= 80:
            f.write("🎉 **Phase 8 Requirements Successfully Met!**\n\n")
        elif completion_rate >= 60:
            f.write("⚠️ **Phase 8 Partially Complete** - Some components need attention\n\n")
        else:
            f.write("❌ **Phase 8 Incomplete** - Significant work remaining\n\n")
            
        # Next steps
        f.write("### 🚀 Next Steps\n\n")
        
        if completion_rate >= 80:
            f.write("- Phase 8 objectives achieved\n")
            f.write("- Ready to proceed to Phase 9 (Continual Learning)\n")
            f.write("- Consider additional model export formats if needed\n")
        else:
            incomplete = [name for (key, name) in components if not completion_status.get(key, False)]
            f.write("Complete the following components:\n")
            for comp in incomplete:
                f.write(f"- {comp}\n")
                
        f.write("\n")
        
        f.write("---\n\n")
        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
    def _extract_mean_value(self, metric_dict: Dict[str, Any], default: str = 'N/A') -> str:
        """Extract mean value from metric dictionary with error handling."""
        if isinstance(metric_dict, dict) and 'mean' in metric_dict:
            value = metric_dict['mean']
            if isinstance(value, (int, float)):
                return f"{value:.3f}"
        return default
        
    def generate_summary_csv(self, final_report: Dict[str, Any]) -> Path:
        """Generate CSV summary of all results for external analysis."""
        
        csv_path = self.results_root / "evaluation_summary.csv"
        
        batch_results = final_report.get('batch_results', {})
        results = batch_results.get('results', {})
        
        csv_data = []
        
        for exp_id, result in results.items():
            if result.get('status') == 'failed':
                continue
                
            model_name = exp_id.replace('adaptive_eval_', '').replace('_', ' ').title()
            agg_metrics = result.get('aggregated_metrics', {})
            in_domain = agg_metrics.get('in_domain', {})
            
            row = {
                'experiment_id': exp_id,
                'model_name': model_name,
                'accuracy_mean': self._safe_extract_float(in_domain.get('top1_accuracy', {}), 'mean'),
                'accuracy_std': self._safe_extract_float(in_domain.get('top1_accuracy', {}), 'std'),
                'time_per_batch_mean': self._safe_extract_float(in_domain.get('time_per_batch_median', {}), 'mean'),
                'time_per_batch_std': self._safe_extract_float(in_domain.get('time_per_batch_median', {}), 'std'),
                'memory_peak_mean': self._safe_extract_float(in_domain.get('memory_peak_mb', {}), 'mean'),
                'memory_peak_std': self._safe_extract_float(in_domain.get('memory_peak_mb', {}), 'std'),
                'energy_proxy_mean': self._safe_extract_float(in_domain.get('energy_proxy', {}), 'mean'),
                'energy_proxy_std': self._safe_extract_float(in_domain.get('energy_proxy', {}), 'std'),
                'ece_mean': self._safe_extract_float(in_domain.get('ece', {}), 'mean'),
                'ece_std': self._safe_extract_float(in_domain.get('ece', {}), 'std'),
                'param_count': result.get('param_count', 0),
                'model_size_mb': result.get('model_size_mb', 0.0)
            }
            
            # Add robustness if available
            if 'robustness_summary' in agg_metrics:
                rob_summary = agg_metrics['robustness_summary']
                row['robustness_mean'] = self._safe_extract_float(rob_summary.get('relative_robustness_mean', {}), 'mean')
                row['robustness_std'] = self._safe_extract_float(rob_summary.get('relative_robustness_mean', {}), 'std')
                
            csv_data.append(row)
            
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            logger.info(f"Summary CSV saved to {csv_path}")
        else:
            logger.warning("No data available for CSV generation")
            
        return csv_path
        
    def _safe_extract_float(self, metric_dict: Dict[str, Any], key: str) -> float:
        """Safely extract float value from metric dictionary."""
        if isinstance(metric_dict, dict) and key in metric_dict:
            value = metric_dict[key]
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test report generator
    generator = ReportGenerator("./test_reports")
    
    # Mock final report
    mock_report = {
        'evaluation_summary': {
            'start_time': '2024-01-01T10:00:00',
            'end_time': '2024-01-01T11:00:00',
            'total_duration_seconds': 3600,
            'total_experiments': 3,
            'successful_experiments': 3,
            'seeds_evaluated': [17, 42, 1337],
            'quick_mode': True
        },
        'batch_results': {
            'results': {
                'adaptive_eval_baseline': {
                    'status': 'completed',
                    'aggregated_metrics': {
                        'in_domain': {
                            'top1_accuracy': {'mean': 85.5, 'std': 1.2},
                            'time_per_batch_median': {'mean': 0.05, 'std': 0.005},
                            'ece': {'mean': 0.08, 'std': 0.01}
                        }
                    },
                    'param_count': 1000000,
                    'model_size_mb': 4.0
                }
            }
        },
        'analysis_results': {},
        'recommendations': {
            'quality_max_mode': {
                'recommended_model': 'Baseline',
                'accuracy_score': 85.5,
                'rationale': 'Highest accuracy'
            }
        },
        'phase8_completion_status': {
            'in_domain_evaluation': True,
            'ood_evaluation': True,
            'cost_analysis': False,
            'comprehensive_dashboard': True
        }
    }
    
    # Generate dashboard
    dashboard_path = generator.generate_comprehensive_dashboard(mock_report)
    print(f"Test dashboard generated at {dashboard_path}")
    
    # Generate CSV
    csv_path = generator.generate_summary_csv(mock_report)
    print(f"Test CSV generated at {csv_path}")