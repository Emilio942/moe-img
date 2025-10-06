import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import json
import numpy as np
from optim.analysis import (
    MetaOptimizerLogger, 
    MetaOptimizerAnalyzer, 
    MetaOptimizerLog,
    create_demo_comparison
)


def test_meta_optimizer_log():
    """Test MetaOptimizerLog dataclass."""
    log_entry = MetaOptimizerLog(
        timestamp=1234567890.0,
        step=10,
        epoch=1,
        loss=0.5,
        accuracy=0.8,
        lr=1e-3,
        weight_decay=1e-4,
        meta_optimizer_type="test"
    )
    
    assert log_entry.timestamp == 1234567890.0
    assert log_entry.step == 10
    assert log_entry.loss == 0.5
    assert log_entry.meta_optimizer_type == "test"


def test_meta_optimizer_logger():
    """Test MetaOptimizerLogger functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test_log.jsonl"
        
        logger = MetaOptimizerLogger(str(log_file))
        
        # Log some steps
        for i in range(5):
            logger.log_step(
                step=i,
                epoch=0,
                loss=1.0 - i * 0.1,
                accuracy=0.5 + i * 0.1,
                optimizer_state={'lr': 1e-3, 'weight_decay': 1e-4, 'grad_norm': 2.0},
                meta_optimizer_type="test",
                action_taken=f"action_{i}" if i % 2 == 0 else None,
                reward=0.1 * i if i > 0 else None,
                costs={'time': 50.0 + i, 'memory': 100.0, 'energy': 25.0}
            )
        
        # Check logs were recorded
        assert len(logger.logs) == 5
        assert logger.logs[0].loss == 1.0
        assert logger.logs[-1].loss == 0.6
        
        # Check file was written
        assert log_file.exists()
        
        # Test DataFrame conversion
        df = logger.get_dataframe()
        assert len(df) == 5
        assert 'datetime' in df.columns
        assert df['loss'].iloc[0] == 1.0
        assert df['meta_optimizer_type'].iloc[0] == "test"


def test_logger_load_from_file():
    """Test loading logger from existing file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test_load.jsonl"
        
        # Create initial logger and write some data
        logger1 = MetaOptimizerLogger(str(log_file))
        logger1.log_step(
            step=0, epoch=0, loss=1.0, accuracy=0.5,
            optimizer_state={'lr': 1e-3, 'weight_decay': 1e-4},
            meta_optimizer_type="test1"
        )
        logger1.log_step(
            step=1, epoch=0, loss=0.9, accuracy=0.6,
            optimizer_state={'lr': 1e-3, 'weight_decay': 1e-4},
            meta_optimizer_type="test1"
        )
        
        # Load from file
        logger2 = MetaOptimizerLogger.load_from_file(str(log_file))
        
        assert len(logger2.logs) == 2
        assert logger2.logs[0].loss == 1.0
        assert logger2.logs[1].loss == 0.9
        assert logger2.logs[0].meta_optimizer_type == "test1"


def test_meta_optimizer_analyzer():
    """Test MetaOptimizerAnalyzer functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = MetaOptimizerAnalyzer(str(temp_dir))
        
        # Create test loggers with sample data
        logger1 = MetaOptimizerLogger()
        logger2 = MetaOptimizerLogger()
        
        # Add sample data
        steps = range(0, 20, 2)
        for i, step in enumerate(steps):
            # Logger 1 - baseline
            logger1.log_step(
                step=step, epoch=step//10, loss=1.0 - i*0.05, accuracy=0.5 + i*0.03,
                optimizer_state={'lr': 1e-3, 'weight_decay': 1e-4},
                meta_optimizer_type="baseline",
                costs={'time': 50.0, 'memory': 100.0, 'energy': 25.0}
            )
            
            # Logger 2 - meta-optimizer with varying LR
            lr = 1e-3 * (0.9 ** (i // 3))  # Decay every 3 steps
            logger2.log_step(
                step=step, epoch=step//10, loss=1.0 - i*0.06, accuracy=0.5 + i*0.035,
                optimizer_state={'lr': lr, 'weight_decay': 1e-4},
                meta_optimizer_type="meta",
                costs={'time': 52.0, 'memory': 102.0, 'energy': 26.0}
            )
        
        loggers = {"Baseline": logger1, "Meta-Optimizer": logger2}
        
        # Test comparison table generation
        comparison_df = analyzer.generate_comparison_table(loggers)
        
        assert len(comparison_df) == 2
        assert "Meta-Optimizer" in comparison_df.columns
        assert "Final Loss" in comparison_df.columns
        assert "LR Adaptations" in comparison_df.columns
        
        # Check that meta-optimizer shows more LR adaptations
        baseline_lr_adaptations = int(comparison_df[comparison_df['Meta-Optimizer'] == 'Baseline']['LR Adaptations'].iloc[0])
        meta_lr_adaptations = int(comparison_df[comparison_df['Meta-Optimizer'] == 'Meta-Optimizer']['LR Adaptations'].iloc[0])
        
        assert meta_lr_adaptations > baseline_lr_adaptations
        
        # Test that plots can be generated (files should be created)
        analyzer.plot_hyperparameter_timeline(loggers)
        analyzer.plot_cost_analysis(loggers)
        
        # Check files were created
        timeline_path = Path(temp_dir) / "hyperparameter_timeline.png"
        cost_path = Path(temp_dir) / "cost_analysis.png"
        
        assert timeline_path.exists()
        assert cost_path.exists()


def test_overhead_measurement():
    """Test overhead measurement functionality."""
    
    def baseline_func():
        """Simulate baseline operation."""
        # Simulate some computation
        x = np.random.rand(100, 100)
        np.dot(x, x.T)
    
    def meta_optimizer_func():
        """Simulate meta-optimizer operation (slightly more expensive)."""
        # Simulate baseline + meta-optimizer overhead
        baseline_func()
        # Additional meta-optimizer computation
        np.random.rand(10, 10)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = MetaOptimizerAnalyzer(str(temp_dir))
        
        overhead_stats = analyzer.measure_overhead(
            meta_optimizer_func=meta_optimizer_func,
            baseline_func=baseline_func,
            num_iterations=20  # Small number for testing
        )
        
        assert 'baseline_avg_ms' in overhead_stats
        assert 'meta_optimizer_avg_ms' in overhead_stats
        assert 'overhead_ms' in overhead_stats
        assert 'overhead_percent' in overhead_stats
        
        # Meta-optimizer should have some overhead
        assert overhead_stats['overhead_ms'] >= 0
        assert overhead_stats['overhead_percent'] >= 0


def test_full_report_generation():
    """Test full report generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = MetaOptimizerAnalyzer(str(temp_dir))
        
        # Create minimal test data
        logger = MetaOptimizerLogger()
        for i in range(5):
            logger.log_step(
                step=i, epoch=0, loss=1.0 - i*0.2, accuracy=0.2 + i*0.15,
                optimizer_state={'lr': 1e-3, 'weight_decay': 1e-4},
                meta_optimizer_type="test"
            )
        
        loggers = {"Test Meta-Optimizer": logger}
        
        # Mock overhead stats
        overhead_stats = {
            'baseline_avg_ms': 50.0,
            'meta_optimizer_avg_ms': 52.0,
            'overhead_ms': 2.0,
            'overhead_percent': 4.0,
            'baseline_std_ms': 1.0,
            'meta_optimizer_std_ms': 1.1
        }
        
        # Generate report
        report_content = analyzer.generate_full_report(loggers, overhead_stats)
        
        # Check report content
        assert "Meta-Optimizer Analysis Report" in report_content
        assert "Computational Overhead Analysis" in report_content
        assert "4.0%" in report_content  # Overhead percentage
        
        # Check files were created
        report_path = Path(temp_dir) / "metaopt_analysis.md"
        assert report_path.exists()
        
        # Check report file content
        with open(report_path, 'r') as f:
            file_content = f.read()
        
        assert "Meta-Optimizer Analysis Report" in file_content
        assert "Computational Overhead Analysis" in file_content


def test_demo_comparison():
    """Test demo comparison function (should run without errors)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to temp directory to avoid creating files in project
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Run demo (should not raise exceptions)
            create_demo_comparison()
            
            # Check that demo files were created
            reports_dir = Path("reports/metaopt_analysis")
            assert reports_dir.exists()
            
            analysis_file = reports_dir / "metaopt_analysis.md"
            assert analysis_file.exists()
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    # Run basic tests
    test_meta_optimizer_log()
    test_meta_optimizer_logger()
    test_logger_load_from_file()
    test_meta_optimizer_analyzer()
    test_overhead_measurement()
    test_full_report_generation()
    print("All analysis tests passed!")