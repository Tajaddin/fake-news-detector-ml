"""
Unified training script for Fake News Detection models
Handles both baseline and transformer models
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from config import (
    MODELS_DIR, 
    EXPERIMENTS_DIR,
    RANDOM_SEED,
    RUNS_CSV,
    BASELINE_CONFIG,
    TRANSFORMER_CONFIG,
    TRAINING_CONFIG
)
from utils import (
    set_seed, 
    log_experiment,
    display_metrics,
    save_checkpoint,
    calculate_class_weights,
    EarlyStopping,
    console
)
from data_io import DataLoader
from features import create_baseline_features, create_advanced_features
from models_baseline import BaselineModel, ModelEnsemble


class ModelTrainer:
    """Unified trainer for all model types"""
    
    def __init__(
        self,
        model_type: str,
        experiment_name: str = "fake_news_detection",
        use_metadata: bool = False,
        seed: int = RANDOM_SEED
    ):
        """
        Initialize trainer
        
        Args:
            model_type: Type of model to train
            experiment_name: Name for experiment logging
            use_metadata: Whether to use metadata features
            seed: Random seed
        """
        self.model_type = model_type
        self.experiment_name = experiment_name
        self.use_metadata = use_metadata
        self.seed = seed
        self.console = Console()
        
        set_seed(seed)
        
        # Determine model category
        self.is_transformer = model_type in ['distilbert', 'bert', 'roberta']
        self.is_baseline = model_type in ['logistic_regression', 'svm', 
                                          'naive_bayes', 'complement_nb']
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def load_data(self) -> tuple:
        """Load and prepare data"""
        self.console.print("[cyan]Loading data...[/cyan]")
        
        loader = DataLoader()
        train_df, val_df, test_df = loader.load_processed_data()
        
        # Check for preprocessed versions
        if self.is_transformer:
            # Try to load minimally preprocessed data for transformers
            for suffix in ['_minimal', '_processed', '']:
                try:
                    train_path = Path(f"data/processed/train_processed{suffix}.csv")
                    if train_path.exists():
                        train_df = pd.read_csv(train_path)
                        val_df = pd.read_csv(f"data/processed/val_processed{suffix}.csv")
                        test_df = pd.read_csv(f"data/processed/test_processed{suffix}.csv")
                        self.console.print(f"  Loaded preprocessed data: {suffix}")
                        break
                except:
                    pass
        
        self.console.print(f"  Train: {len(train_df):,} samples")
        self.console.print(f"  Val: {len(val_df):,} samples")
        self.console.print(f"  Test: {len(test_df):,} samples")
        
        return train_df, val_df, test_df
    
    def prepare_features(self, train_df, val_df, test_df):
        """Prepare features based on model type"""
        
        if self.is_baseline:
            # Use feature extraction for baseline models
            self.console.print("[cyan]Extracting features...[/cyan]")
            
            if self.use_metadata:
                feature_extractor = create_advanced_features()
                self.console.print("  Using TF-IDF + metadata features")
            else:
                feature_extractor = create_baseline_features()
                self.console.print("  Using TF-IDF features only")
            
            X_train = feature_extractor.fit_transform(train_df)
            X_val = feature_extractor.transform(val_df)
            X_test = feature_extractor.transform(test_df)
            
            self.console.print(f"  Feature dimensions: {X_train.shape[1]:,}")
            
            # Save feature extractor
            feature_path = MODELS_DIR / f"feature_extractor_{self.model_type}.pkl"
            feature_extractor.save(feature_path)
            
            return X_train, X_val, X_test, feature_extractor
        
        else:
            # For transformers, return DataFrames directly
            return train_df, val_df, test_df, None
    
    def train_baseline(self, X_train, y_train, X_val, y_val):
        """Train baseline model"""
        model = BaselineModel(
            model_type=self.model_type,
            random_state=self.seed,
            class_weight='balanced'
        )
        
        # Train with hyperparameter tuning
        model.train(
            X_train, y_train,
            X_val, y_val,
            tune_hyperparams=True,
            cv_folds=5
        )
        
        return model
    
    def train_transformer(self, train_df, val_df):
        """Train transformer model (placeholder - implement in models_transformer.py)"""
        self.console.print("[yellow]Transformer training will be implemented in Phase 3[/yellow]")
        return None
    
    def train(self):
        """Main training pipeline"""
        start_time = time.time()
        
        # Print header
        self.console.print(f"\n{'='*60}")
        self.console.print(f"[bold cyan]Training {self.model_type.upper()} Model[/bold cyan]")
        self.console.print(f"Experiment: {self.experiment_name}")
        self.console.print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.console.print(f"{'='*60}\n")
        
        # Load data
        train_df, val_df, test_df = self.load_data()
        
        # Prepare features
        X_train, X_val, X_test, feature_extractor = self.prepare_features(
            train_df, val_df, test_df
        )
        
        # Get labels
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        y_test = test_df['label'].values
        
        # Calculate class weights
        class_weights = calculate_class_weights(y_train)
        
        # Train model
        if self.is_baseline:
            model = self.train_baseline(X_train, y_train, X_val, y_val)
        else:
            model = self.train_transformer(train_df, val_df)
        
        if model is None:
            return None
        
        # Evaluate on all sets
        self.console.print("\n[bold]Evaluating model...[/bold]")
        
        train_metrics = model.evaluate(X_train, y_train)
        val_metrics = model.evaluate(X_val, y_val)
        test_metrics = model.evaluate(X_test, y_test)
        
        # Display results
        self.console.print("\n[bold]Training Set Performance:[/bold]")
        display_metrics(train_metrics, title="Train Metrics")
        
        self.console.print("\n[bold]Validation Set Performance:[/bold]")
        display_metrics(val_metrics, title="Validation Metrics")
        
        self.console.print("\n[bold]Test Set Performance:[/bold]")
        display_metrics(test_metrics, title="Test Metrics")
        
        # Save model
        model_path = MODELS_DIR / f"{self.model_type}_{self.seed}.pkl"
        model.save(model_path)
        
        # Log experiment
        hyperparams = model.best_params_ if hasattr(model, 'best_params_') else {}
        hyperparams['use_metadata'] = self.use_metadata
        
        log_experiment(
            experiment_name=self.experiment_name,
            model_name=self.model_type,
            features='tfidf+metadata' if self.use_metadata else 'tfidf',
            metrics=test_metrics,
            hyperparams=hyperparams,
            seed=self.seed,
            csv_path=RUNS_CSV
        )
        
        # Feature importance for interpretable models
        if self.is_baseline and feature_extractor:
            self.console.print("\n[bold]Top Feature Importance:[/bold]")
            feature_names = feature_extractor.get_feature_names()
            importance_df = model.get_feature_importance(feature_names, top_k=15)
            
            if not importance_df.empty:
                for _, row in importance_df.head(10).iterrows():
                    if 'coefficient' in importance_df.columns:
                        self.console.print(f"  {row['feature']:40s} {row['coefficient']:+.4f}")
                    elif 'importance' in importance_df.columns:
                        self.console.print(f"  {row['feature']:40s} {row['importance']:.4f}")
        
        # Training summary
        training_time = time.time() - start_time
        self.console.print(f"\n{'='*60}")
        self.console.print("[bold green]✨ Training Complete![/bold green]")
        self.console.print(f"Total time: {training_time:.2f} seconds")
        self.console.print(f"Model saved: {model_path}")
        self.console.print(f"{'='*60}\n")
        
        return model, test_metrics
    
    def cross_validate(self, n_runs: int = 3):
        """Run multiple training runs with different seeds"""
        self.console.print(f"\n[bold]Running {n_runs}-fold cross-validation...[/bold]")
        
        all_metrics = []
        
        for run in range(n_runs):
            seed = RANDOM_SEED + run
            self.console.print(f"\n[cyan]Run {run+1}/{n_runs} (seed={seed})[/cyan]")
            
            # Create new trainer with different seed
            trainer = ModelTrainer(
                model_type=self.model_type,
                experiment_name=f"{self.experiment_name}_run{run+1}",
                use_metadata=self.use_metadata,
                seed=seed
            )
            
            # Train and get metrics
            _, metrics = trainer.train()
            if metrics:
                all_metrics.append(metrics)
        
        # Compute statistics
        if all_metrics:
            self.console.print("\n[bold]Cross-Validation Results:[/bold]")
            
            # Calculate mean and std for each metric
            metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            
            for metric in metric_names:
                values = [m[metric] for m in all_metrics if metric in m]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    self.console.print(f"  {metric.capitalize():10s}: "
                                     f"{mean_val:.4f} (+/- {std_val:.4f})")
            
            # Save CV results
            cv_results = {
                'model_type': self.model_type,
                'n_runs': n_runs,
                'metrics': all_metrics,
                'summary': {
                    metric: {
                        'mean': np.mean([m[metric] for m in all_metrics if metric in m]),
                        'std': np.std([m[metric] for m in all_metrics if metric in m])
                    }
                    for metric in metric_names
                }
            }
            
            cv_path = EXPERIMENTS_DIR / f"cv_results_{self.model_type}.json"
            with open(cv_path, 'w') as f:
                json.dump(cv_results, f, indent=2, default=str)
            
            self.console.print(f"\n[green]✓ CV results saved to {cv_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Train fake news detection models")
    
    parser.add_argument('--model', type=str, default='logistic_regression',
                       choices=['logistic_regression', 'svm', 'naive_bayes', 
                               'complement_nb', 'random_forest',
                               'distilbert', 'bert', 'roberta'],
                       help='Model type to train')
    
    parser.add_argument('--metadata', action='store_true',
                       help='Include metadata features')
    
    parser.add_argument('--experiment', type=str, default='baseline',
                       help='Experiment name for tracking')
    
    parser.add_argument('--cv', type=int, default=0,
                       help='Number of cross-validation runs (0 for single run)')
    
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ModelTrainer(
        model_type=args.model,
        experiment_name=args.experiment,
        use_metadata=args.metadata,
        seed=args.seed
    )
    
    # Run training
    if args.cv > 0:
        trainer.cross_validate(n_runs=args.cv)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
