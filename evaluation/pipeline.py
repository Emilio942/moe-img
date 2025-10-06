"""
Evaluation Pipeline and Orchestration for Adaptive Evaluation Suite
================================================================

Main pipeline for running comprehensive model evaluation across multiple
seeds, datasets, and configurations with robust logging and error handling.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

from .datasets import EvaluationDatasets, create_ood_splits
from .metrics import (
    CoreMetrics, CostMetrics, RobustnessMetrics, CalibrationMetrics,
    MetricsResult, TemperatureScaling, compute_model_size, get_system_info
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single evaluation experiment."""
    
    # Experiment identification
    experiment_id: str
    model_name: str
    checkpoint_path: str
    
    # Evaluation settings
    seeds: List[int]
    batch_size: int
    max_batches: Optional[int] = None  # For quick evaluation
    
    # Dataset configuration
    evaluate_ood: bool = True
    max_corruptions: Optional[int] = None  # Limit OOD corruptions
    
    # Calibration settings
    apply_temperature_scaling: bool = True
    temperature_lr: float = 0.01
    
    # Cost measurement settings
    warmup_batches: int = 2
    measure_energy: bool = True
    
    # Output configuration
    save_detailed_results: bool = True
    save_logits: bool = False  # For calibration analysis
    
    # Metadata
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**data)
        
    def compute_hash(self) -> str:
        """Compute hash for config reproducibility."""
        # Exclude variable fields from hash
        hash_data = {
            'model_name': self.model_name,
            'seeds': sorted(self.seeds),
            'batch_size': self.batch_size,
            'evaluate_ood': self.evaluate_ood,
            'max_corruptions': self.max_corruptions
        }
        hash_input = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]


class EvaluationPipeline:
    """
    Main evaluation pipeline for comprehensive model assessment.
    Handles single-model evaluation across multiple seeds and datasets.
    """
    
    def __init__(self, 
                 data_root: str = "./data",
                 results_root: str = "./reports",
                 device: Optional[torch.device] = None):
        self.data_root = Path(data_root)
        self.results_root = Path(results_root)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize datasets manager
        self.datasets_manager = None
        
        # Results storage
        self.experiment_results = {}
        
        logger.info(f"EvaluationPipeline initialized on device: {self.device}")
        
    def setup_datasets(self, seed: int = 42, batch_size: int = 256):
        """Initialize datasets manager with specific configuration."""
        self.datasets_manager = EvaluationDatasets(
            data_root=str(self.data_root),
            batch_size=batch_size,
            seed=seed
        )
        logger.info(f"Datasets configured with seed {seed}, batch_size {batch_size}")
        
    def load_model(self, checkpoint_path: str) -> nn.Module:
        """
        Load model from checkpoint with error handling.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Loaded PyTorch model
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
                
            # Load checkpoint
            logger.info(f"Loading model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract model (handle different checkpoint formats)
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'state_dict' in checkpoint:
                # Need model architecture - this is a limitation
                # In practice, you'd have model creation logic here
                raise NotImplementedError("state_dict loading requires model architecture")
            else:
                # Assume checkpoint is the model directly
                model = checkpoint
                
            model = model.to(self.device)
            model.eval()
            
            # Verify model is working
            with torch.no_grad():
                test_input = torch.randn(1, 3, 32, 32).to(self.device)
                test_output = model(test_input)
                logger.info(f"Model loaded successfully, output shape: {test_output.shape}")
                
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {checkpoint_path}: {e}")
            raise
            
    def evaluate_single_seed(self,
                           model: nn.Module,
                           config: ExperimentConfig,
                           seed: int) -> Dict[str, Any]:
        """
        Evaluate model on all datasets for a single seed.
        
        Args:
            model: PyTorch model to evaluate
            config: Experiment configuration
            seed: Random seed for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating seed {seed} for {config.experiment_id}")
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
        # Setup datasets with this seed
        self.setup_datasets(seed=seed, batch_size=config.batch_size)
        
        # Get evaluation splits
        evaluation_splits = create_ood_splits(
            self.datasets_manager, 
            max_corruptions=config.max_corruptions
        )
        
        results = {
            'seed': seed,
            'config_hash': config.compute_hash(),
            'timestamp': datetime.now().isoformat(),
            'system_info': get_system_info(),
        }
        
        # Model size metrics
        param_count, model_size_mb = compute_model_size(model)
        results.update({
            'param_count': param_count,
            'model_size_mb': model_size_mb
        })
        
        try:
            # 1. In-domain evaluation
            logger.info(f"Evaluating in-domain dataset (seed {seed})")
            in_domain_results = self._evaluate_single_dataset(
                model, evaluation_splits['in_domain']['loader'],
                dataset_name="in_domain", config=config
            )
            results['in_domain'] = in_domain_results
            
            # 2. Validation set evaluation (for temperature scaling)
            if config.apply_temperature_scaling:
                logger.info(f"Evaluating validation set for calibration (seed {seed})")
                val_results = self._evaluate_single_dataset(
                    model, evaluation_splits['validation']['loader'],
                    dataset_name="validation", config=config,
                    collect_logits=True
                )
                results['validation'] = val_results
                
                # Fit temperature scaling
                if 'logits' in val_results and 'targets' in val_results:
                    temp_scaler = TemperatureScaling()
                    logits_tensor = torch.tensor(val_results['logits'])
                    targets_tensor = torch.tensor(val_results['targets'])
                    
                    temperature = temp_scaler.fit(
                        logits_tensor, targets_tensor, 
                        lr=config.temperature_lr
                    )
                    results['temperature'] = temperature
                else:
                    logger.warning("Could not fit temperature scaling - missing logits/targets")
                    results['temperature'] = 1.0
            else:
                results['temperature'] = 1.0
                
            # 3. OOD evaluation
            if config.evaluate_ood:
                logger.info(f"Evaluating {len(evaluation_splits['ood_datasets'])} OOD datasets (seed {seed})")
                ood_results = {}
                
                for ood_name, (ood_loader, ood_meta) in evaluation_splits['ood_datasets'].items():
                    try:
                        ood_result = self._evaluate_single_dataset(
                            model, ood_loader,
                            dataset_name=ood_name, config=config,
                            temperature=results.get('temperature', 1.0)
                        )
                        ood_results[ood_name] = ood_result
                        
                    except Exception as e:
                        logger.error(f"Failed to evaluate OOD dataset {ood_name}: {e}")
                        ood_results[ood_name] = {'error': str(e)}
                        
                results['ood_datasets'] = ood_results
                
                # Compute robustness summary
                if 'in_domain' in results:
                    robustness_metrics = RobustnessMetrics()
                    robustness_metrics.update_in_domain(results['in_domain']['top1_accuracy'])
                    
                    for ood_name, ood_result in ood_results.items():
                        if 'error' not in ood_result:
                            robustness_metrics.update_corruption(
                                ood_name,
                                ood_result['top1_accuracy'],
                                ood_result.get('confidence_mean', 0.0)
                            )
                            
                    results['robustness_summary'] = robustness_metrics.compute_robustness_scores()
                    
        except Exception as e:
            logger.error(f"Error during seed {seed} evaluation: {e}")
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
            
        return results
        
    def _evaluate_single_dataset(self,
                               model: nn.Module,
                               dataloader: DataLoader,
                               dataset_name: str,
                               config: ExperimentConfig,
                               temperature: float = 1.0,
                               collect_logits: bool = False) -> Dict[str, Any]:
        """
        Evaluate model on a single dataset.
        
        Args:
            model: PyTorch model
            dataloader: Dataset loader
            dataset_name: Name for logging
            config: Experiment configuration
            temperature: Temperature for scaling (if > 1.0)
            collect_logits: Whether to collect logits for calibration
            
        Returns:
            Dictionary with evaluation results
        """
        model.eval()
        
        # Initialize metrics
        core_metrics = CoreMetrics()
        cost_metrics = CostMetrics(warmup_batches=config.warmup_batches)
        cal_metrics = CalibrationMetrics()
        
        # Storage for logits if needed
        all_logits = []
        all_targets = []
        
        total_batches = len(dataloader)
        max_batches = config.max_batches or total_batches
        max_batches = min(max_batches, total_batches)
        
        logger.info(f"Evaluating {dataset_name}: {max_batches}/{total_batches} batches")
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                    
                # Move to device
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Measure batch cost
                with cost_metrics.measure_batch():
                    # Forward pass
                    outputs = model(images)
                    
                    # Apply temperature scaling if needed
                    if temperature != 1.0:
                        outputs = outputs / temperature
                        
                    # Compute loss
                    loss = nn.functional.cross_entropy(outputs, targets)
                    
                # Update metrics
                core_metrics.update(outputs, targets, loss)
                cal_metrics.update(outputs, targets)
                
                # Collect logits if requested
                if collect_logits:
                    all_logits.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    
                # Progress logging
                if (batch_idx + 1) % 50 == 0 or batch_idx == max_batches - 1:
                    logger.debug(f"{dataset_name}: batch {batch_idx + 1}/{max_batches}")
                    
        # Compute final metrics
        core_results = core_metrics.compute()
        cost_results = cost_metrics.compute()
        ece, mce = cal_metrics.compute_ece_mce()
        
        # Combine results
        results = {
            **core_results,
            **cost_results,
            'ece': ece,
            'mce': mce,
            'num_batches_evaluated': max_batches,
            'total_batches_available': total_batches
        }
        
        # Add logits if collected
        if collect_logits:
            results['logits'] = np.concatenate(all_logits, axis=0).tolist()
            results['targets'] = np.concatenate(all_targets, axis=0).tolist()
            
        return results
        
    def run_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Run complete experiment across all seeds.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Aggregated results across seeds
        """
        logger.info(f"Starting experiment: {config.experiment_id}")
        start_time = time.time()
        
        # Load model
        try:
            model = self.load_model(config.checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to load model for {config.experiment_id}: {e}")
            return {
                'experiment_id': config.experiment_id,
                'error': f"Model loading failed: {e}",
                'status': 'failed'
            }
            
        # Run evaluation for each seed
        seed_results = []
        failed_seeds = []
        
        for seed in config.seeds:
            try:
                seed_result = self.evaluate_single_seed(model, config, seed)
                seed_results.append(seed_result)
                
                if 'error' in seed_result:
                    failed_seeds.append(seed)
                    logger.warning(f"Seed {seed} completed with errors")
                else:
                    logger.info(f"Seed {seed} completed successfully")
                    
            except Exception as e:
                logger.error(f"Critical failure for seed {seed}: {e}")
                failed_seeds.append(seed)
                seed_results.append({
                    'seed': seed,
                    'error': str(e),
                    'critical_failure': True
                })
                
        # Aggregate results across seeds
        aggregated_results = self._aggregate_seed_results(seed_results, config)
        
        # Add experiment metadata
        aggregated_results.update({
            'experiment_id': config.experiment_id,
            'config': config.to_dict(),
            'total_seeds': len(config.seeds),
            'successful_seeds': len(config.seeds) - len(failed_seeds),
            'failed_seeds': failed_seeds,
            'experiment_duration': time.time() - start_time,
            'status': 'completed' if not failed_seeds else 'partial_failure'
        })
        
        # Save results
        if config.save_detailed_results:
            self._save_experiment_results(aggregated_results)
            
        logger.info(f"Experiment {config.experiment_id} completed in {aggregated_results['experiment_duration']:.2f}s")
        return aggregated_results
        
    def _aggregate_seed_results(self, 
                              seed_results: List[Dict[str, Any]],
                              config: ExperimentConfig) -> Dict[str, Any]:
        """Aggregate results across multiple seeds."""
        
        # Filter successful results
        successful_results = [r for r in seed_results if 'error' not in r or not r.get('critical_failure', False)]
        
        if not successful_results:
            return {
                'aggregated_metrics': {},
                'error': 'All seeds failed'
            }
            
        aggregated = {
            'seed_results': seed_results,
            'aggregated_metrics': {}
        }
        
        # Aggregate in-domain results
        if all('in_domain' in r for r in successful_results):
            in_domain_agg = self._aggregate_metrics([r['in_domain'] for r in successful_results])
            aggregated['aggregated_metrics']['in_domain'] = in_domain_agg
            
        # Aggregate OOD results
        if config.evaluate_ood and all('ood_datasets' in r for r in successful_results):
            ood_aggregated = {}
            
            # Get all OOD dataset names
            all_ood_names = set()
            for r in successful_results:
                if 'ood_datasets' in r:
                    all_ood_names.update(r['ood_datasets'].keys())
                    
            # Aggregate each OOD dataset
            for ood_name in all_ood_names:
                ood_results_for_name = []
                for r in successful_results:
                    if ('ood_datasets' in r and 
                        ood_name in r['ood_datasets'] and 
                        'error' not in r['ood_datasets'][ood_name]):
                        ood_results_for_name.append(r['ood_datasets'][ood_name])
                        
                if ood_results_for_name:
                    ood_aggregated[ood_name] = self._aggregate_metrics(ood_results_for_name)
                    
            aggregated['aggregated_metrics']['ood_datasets'] = ood_aggregated
            
        # Aggregate robustness summary
        if all('robustness_summary' in r for r in successful_results):
            robustness_results = [r['robustness_summary'] for r in successful_results]
            aggregated['aggregated_metrics']['robustness_summary'] = self._aggregate_metrics(robustness_results)
            
        return aggregated
        
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics with mean and std computation."""
        if not metrics_list:
            return {}
            
        aggregated = {}
        
        # Get all metric names
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
            
        # Compute statistics for each metric
        for key in all_keys:
            values = [m.get(key, 0.0) for m in metrics_list if key in m]
            
            if values:
                aggregated[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
            else:
                aggregated[key] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }
                
        return aggregated
        
    def _save_experiment_results(self, results: Dict[str, Any]):
        """Save experiment results to files."""
        experiment_id = results['experiment_id']
        
        # Create results directory
        results_dir = self.results_root / "adaptive_evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed JSON results
        json_path = results_dir / f"{experiment_id}_detailed.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Detailed results saved to {json_path}")
        
        # Save summary results (without detailed logits)
        summary_results = results.copy()
        
        # Remove large data fields
        if 'seed_results' in summary_results:
            for seed_result in summary_results['seed_results']:
                if 'validation' in seed_result and 'logits' in seed_result['validation']:
                    del seed_result['validation']['logits']
                if 'validation' in seed_result and 'targets' in seed_result['validation']:
                    del seed_result['validation']['targets']
                    
        summary_path = results_dir / f"{experiment_id}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_results, f, indent=2, default=str)
            
        logger.info(f"Summary results saved to {summary_path}")


class EvaluationRunner:
    """
    High-level runner for multiple experiments with parallelization support.
    """
    
    def __init__(self, 
                 data_root: str = "./data",
                 results_root: str = "./reports",
                 max_workers: int = 1):
        self.data_root = data_root
        self.results_root = Path(results_root)
        self.max_workers = max_workers
        
        # Ensure results directory exists
        self.results_root.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"EvaluationRunner initialized with max_workers={max_workers}")
        
    def run_experiments(self, 
                       configs: List[ExperimentConfig],
                       parallel: bool = False) -> Dict[str, Any]:
        """
        Run multiple experiments with optional parallelization.
        
        Args:
            configs: List of experiment configurations
            parallel: Whether to run experiments in parallel
            
        Returns:
            Dictionary with all experiment results
        """
        logger.info(f"Running {len(configs)} experiments (parallel={parallel})")
        
        all_results = {}
        failed_experiments = []
        
        if parallel and self.max_workers > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_config = {
                    executor.submit(self._run_single_experiment, config): config
                    for config in configs
                }
                
                for future in future_to_config:
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        all_results[config.experiment_id] = result
                        
                        if result.get('status') == 'failed':
                            failed_experiments.append(config.experiment_id)
                            
                    except Exception as e:
                        logger.error(f"Experiment {config.experiment_id} failed: {e}")
                        failed_experiments.append(config.experiment_id)
                        all_results[config.experiment_id] = {
                            'experiment_id': config.experiment_id,
                            'error': str(e),
                            'status': 'failed'
                        }
        else:
            # Sequential execution
            for config in configs:
                try:
                    result = self._run_single_experiment(config)
                    all_results[config.experiment_id] = result
                    
                    if result.get('status') == 'failed':
                        failed_experiments.append(config.experiment_id)
                        
                except Exception as e:
                    logger.error(f"Experiment {config.experiment_id} failed: {e}")
                    failed_experiments.append(config.experiment_id)
                    all_results[config.experiment_id] = {
                        'experiment_id': config.experiment_id,
                        'error': str(e),
                        'status': 'failed'
                    }
                    
        # Generate summary
        summary = {
            'total_experiments': len(configs),
            'successful_experiments': len(configs) - len(failed_experiments),
            'failed_experiments': failed_experiments,
            'results': all_results,
            'execution_timestamp': datetime.now().isoformat()
        }
        
        # Save batch summary
        self._save_batch_summary(summary)
        
        logger.info(f"Batch evaluation completed: "
                   f"{summary['successful_experiments']}/{summary['total_experiments']} successful")
        
        return summary
        
    def _run_single_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment in isolation."""
        pipeline = EvaluationPipeline(
            data_root=self.data_root,
            results_root=str(self.results_root)
        )
        
        return pipeline.run_experiment(config)
        
    def _save_batch_summary(self, summary: Dict[str, Any]):
        """Save batch execution summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.results_root / f"batch_summary_{timestamp}.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        logger.info(f"Batch summary saved to {summary_path}")


# Example configuration loading
def load_configs_from_yaml(yaml_path: str) -> List[ExperimentConfig]:
    """Load experiment configurations from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    configs = []
    for exp_data in data['experiments']:
        config = ExperimentConfig.from_dict(exp_data)
        configs.append(config)
        
    logger.info(f"Loaded {len(configs)} experiment configurations from {yaml_path}")
    return configs


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create example configuration
    config = ExperimentConfig(
        experiment_id="test_baseline",
        model_name="MoE_Baseline",
        checkpoint_path="./checkpoints/baseline.ckpt",
        seeds=[42, 1337],
        batch_size=32,
        max_batches=10,  # Quick test
        evaluate_ood=True,
        max_corruptions=5,  # Limited OOD for testing
        description="Test run for pipeline validation"
    )
    
    print(f"Example config: {config.experiment_id}")
    print(f"Config hash: {config.compute_hash()}")
    
    # Test pipeline (would need actual model)
    # pipeline = EvaluationPipeline()
    # result = pipeline.run_experiment(config)