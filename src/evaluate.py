"""
FIXED Evaluation module for Fake News Detection models
Properly loads saved feature extractors without data leakage
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score, matthews_corrcoef, cohen_kappa_score
)
from sklearn.calibration import calibration_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from rich.console import Console
from rich.table import Table

from config import MODELS_DIR, FIGURES_DIR, EXPERIMENTS_DIR, PROCESSED_DATA_DIR
from utils import console, set_seed, display_metrics
from data_io import DataLoader
from features import FeatureExtractor


class ModelEvaluator:
    """Comprehensive model evaluation and analysis"""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        feature_extractor_path: Optional[Path] = None
    ):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to saved model
            feature_extractor_path: Path to saved feature extractor
        """
        self.model = None
        self.feature_extractor = None
        self.console = Console()
        
        if model_path:
            self.load_model(model_path)
        
        if feature_extractor_path:
            self.load_feature_extractor(feature_extractor_path)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette('husl')
    
    def load_model(self, path: Path):
        """Load a saved model"""
        from models_baseline import BaselineModel
        
        if path.suffix == '.pkl':
            # Try loading as baseline model
            try:
                self.model = BaselineModel.load(path)
                self.console.print(f"[green]✓ Loaded model from {path}[/green]")
            except:
                # Try loading as joblib object
                self.model = joblib.load(path)
                self.console.print(f"[green]✓ Loaded model from {path}[/green]")
    
    def load_feature_extractor(self, path: Path):
        """Load a saved feature extractor"""
        self.feature_extractor = FeatureExtractor.load(path)
        self.console.print(f"[green]✓ Loaded feature extractor from {path}[/green]")
    
    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of model performance
        
        Args:
            X: Feature matrix
            y_true: True labels
            model_name: Name for display
        
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Get probabilities if available
        try:
            y_proba = self.model.predict_proba(X)
            has_proba = True
        except:
            y_proba = None
            has_proba = False
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'kappa': cohen_kappa_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Add probability-based metrics if available
        if has_proba and len(np.unique(y_true)) == 2:
            metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
            metrics['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
            metrics['probabilities'] = y_proba
        
        metrics['predictions'] = y_pred
        metrics['model_name'] = model_name
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: Optional[Path] = None
    ):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        
        plt.title(title, fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add percentages
        for i in range(2):
            for j in range(2):
                total = cm.sum()
                percentage = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'{percentage:.1f}%',
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            self.console.print(f"[green]✓ Confusion matrix saved to {save_path}[/green]")
        
        plt.show()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[Path] = None
    ):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
        
        # Fill area under curve
        plt.fill_between(fpr, tpr, alpha=0.3)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=16)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            self.console.print(f"[green]✓ ROC curve saved to {save_path}[/green]")
        
        plt.show()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Precision-Recall Curve",
        save_path: Optional[Path] = None
    ):
        """Plot precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        
        # Plot PR curve
        plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})', linewidth=2)
        
        # Fill area under curve
        plt.fill_between(recall, precision, alpha=0.3)
        
        # Add baseline
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='k', linestyle='--', 
                   label=f'Baseline (prevalence = {baseline:.3f})', alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=16)
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            self.console.print(f"[green]✓ PR curve saved to {save_path}[/green]")
        
        plt.show()
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Calibration Plot",
        save_path: Optional[Path] = None
    ):
        """Plot calibration curve"""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10, strategy='uniform'
        )
        
        plt.figure(figsize=(8, 6))
        
        # Plot calibration curve
        plt.plot(mean_predicted_value, fraction_of_positives, 
                marker='o', linewidth=2, label='Model', markersize=8)
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
        
        # Add histogram
        ax2 = plt.gca().twinx()
        ax2.hist(y_proba, bins=10, alpha=0.3, color='gray', edgecolor='black')
        ax2.set_ylabel('Count', fontsize=12)
        
        plt.xlabel('Mean predicted probability', fontsize=12)
        plt.ylabel('Fraction of positives', fontsize=12)
        plt.title(title, fontsize=16)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            self.console.print(f"[green]✓ Calibration plot saved to {save_path}[/green]")
        
        plt.show()
    
    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[Path] = None
    ) -> Tuple[float, float]:
        """Plot threshold analysis"""
        thresholds = np.linspace(0.1, 0.9, 50)
        metrics_by_threshold = {
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            metrics_by_threshold['precision'].append(
                precision_score(y_true, y_pred, zero_division=0)
            )
            metrics_by_threshold['recall'].append(
                recall_score(y_true, y_pred, zero_division=0)
            )
            metrics_by_threshold['f1'].append(
                f1_score(y_true, y_pred, zero_division=0)
            )
        
        # Find best threshold
        best_idx = np.argmax(metrics_by_threshold['f1'])
        best_threshold = thresholds[best_idx]
        best_f1 = metrics_by_threshold['f1'][best_idx]
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Metrics vs Threshold
        ax1 = axes[0]
        ax1.plot(thresholds, metrics_by_threshold['precision'], label='Precision', linewidth=2)
        ax1.plot(thresholds, metrics_by_threshold['recall'], label='Recall', linewidth=2)
        ax1.plot(thresholds, metrics_by_threshold['f1'], label='F1-Score', linewidth=2)
        ax1.axvline(best_threshold, color='red', linestyle='--', 
                   label=f'Best F1 @ {best_threshold:.2f}', alpha=0.7)
        ax1.set_xlabel('Threshold', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Metrics vs Threshold', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Precision-Recall Trade-off
        ax2 = axes[1]
        ax2.plot(metrics_by_threshold['recall'], metrics_by_threshold['precision'], linewidth=2)
        ax2.scatter([metrics_by_threshold['recall'][best_idx]], 
                   [metrics_by_threshold['precision'][best_idx]], 
                   color='red', s=100, zorder=5)
        
        # Add threshold labels
        for i in range(0, len(thresholds), 10):
            ax2.annotate(f'{thresholds[i]:.2f}', 
                        (metrics_by_threshold['recall'][i], metrics_by_threshold['precision'][i]),
                        fontsize=8, alpha=0.7)
        
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Trade-off', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Threshold Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            self.console.print(f"[green]✓ Threshold analysis saved to {save_path}[/green]")
        
        plt.show()
        
        return best_threshold, best_f1


def main():
    """Main evaluation script - FIXED VERSION"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file')
    parser.add_argument('--all', action='store_true',
                       help='Evaluate all saved models')
    parser.add_argument('--plots', action='store_true',
                       help='Generate evaluation plots')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed report')
    args = parser.parse_args()
    
    # Load test data
    console.print("[cyan]Loading test data...[/cyan]")
    loader = DataLoader()
    _, _, test_df = loader.load_processed_data()
    
    # CRITICAL FIX: Load the correct feature extractor
    if args.model:
        model_path = Path(args.model)
        model_name = model_path.stem.split('_')[0]  # Extract model type
        
        # Look for matching feature extractor
        feature_paths = [
            MODELS_DIR / f"feature_extractor_{model_name}_{model_path.stem.split('_')[-1]}.pkl",
            MODELS_DIR / f"feature_extractor_{model_name}.pkl",
            MODELS_DIR / "feature_extractor.pkl"
        ]
        
        feature_path = None
        for path in feature_paths:
            if path.exists():
                feature_path = path
                break
        
        if feature_path is None:
            # Look for any feature extractor
            feature_files = list(MODELS_DIR.glob("feature_extractor*.pkl"))
            if feature_files:
                feature_path = feature_files[0]
                console.print(f"[yellow]Warning: Using {feature_path.name}[/yellow]")
            else:
                console.print("[red]ERROR: No feature extractor found![/red]")
                console.print("[yellow]You must train a model first to create the feature extractor.[/yellow]")
                console.print("[yellow]Run: python train.py --model logistic_regression[/yellow]")
                exit(1)
        
        # Load feature extractor and transform test data
        console.print(f"[cyan]Loading feature extractor from {feature_path}[/cyan]")
        feature_extractor = FeatureExtractor.load(feature_path)
        
        # CRITICAL: Only transform, never fit on test data!
        X_test = feature_extractor.transform(test_df)
        y_test = test_df['label'].values
        
        console.print(f"[green]✓ Test features extracted: {X_test.shape}[/green]")
        
        # Evaluate single model
        evaluator = ModelEvaluator()
        evaluator.load_model(model_path)
        
        # Evaluate
        metrics = evaluator.evaluate(X_test, y_test, model_name=model_path.stem)
        
        # Display metrics
        console.print("\n[bold]Test Set Performance:[/bold]")
        display_metrics(metrics)
        
        # Check for suspiciously high performance
        if metrics['accuracy'] >= 0.99:
            console.print("\n[yellow]⚠️ WARNING: Near-perfect accuracy detected![/yellow]")
            console.print("[yellow]This suggests either:[/yellow]")
            console.print("[yellow]1. A toy/synthetic dataset[/yellow]")
            console.print("[yellow]2. Data leakage between train and test[/yellow]")
            console.print("[yellow]3. Overfitting on a very small dataset[/yellow]")
            console.print("[yellow]Please investigate your data![/yellow]")
        
        # Generate plots if requested
        if args.plots and 'probabilities' in metrics:
            y_proba = metrics['probabilities'][:, 1]
            
            # All visualization methods...
            evaluator.plot_confusion_matrix(
                metrics['confusion_matrix'],
                title=f"Confusion Matrix - {model_path.stem}",
                save_path=FIGURES_DIR / f"confusion_matrix_{model_path.stem}.png"
            )
            
            evaluator.plot_roc_curve(
                y_test, y_proba,
                title=f"ROC Curve - {model_path.stem}",
                save_path=FIGURES_DIR / f"roc_curve_{model_path.stem}.png"
            )
            
            evaluator.plot_precision_recall_curve(
                y_test, y_proba,
                title=f"Precision-Recall Curve - {model_path.stem}",
                save_path=FIGURES_DIR / f"pr_curve_{model_path.stem}.png"
            )
            
            evaluator.plot_calibration_curve(
                y_test, y_proba,
                title=f"Calibration Plot - {model_path.stem}",
                save_path=FIGURES_DIR / f"calibration_{model_path.stem}.png"
            )
            
            best_threshold, best_f1 = evaluator.plot_threshold_analysis(
                y_test, y_proba,
                save_path=FIGURES_DIR / f"threshold_analysis_{model_path.stem}.png"
            )
            
            console.print(f"\n[green]Optimal threshold: {best_threshold:.3f} (F1={best_f1:.4f})[/green]")
    
    else:
        console.print("[yellow]Please specify --model or --all[/yellow]")
    
    console.print("\n[bold green]✨ Evaluation complete![/bold green]")


if __name__ == "__main__":
    main()
