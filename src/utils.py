"""
Utility functions for the Fake News Detection project
"""

import random
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import csv
from datetime import datetime
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Initialize rich console for pretty printing
console = Console()

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    console.print(f"[green]✓[/green] Random seed set to {seed}")

def setup_logging(name: str = "fake_news", level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        name: Logger name
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    return logger

def log_experiment(
    experiment_name: str,
    model_name: str,
    features: str,
    metrics: Dict[str, float],
    hyperparams: Dict[str, Any],
    seed: int,
    csv_path: Path
):
    """
    Log experiment results to CSV file
    
    Args:
        experiment_name: Name of the experiment
        model_name: Model type (e.g., "logistic_regression", "distilbert")
        features: Feature type used
        metrics: Dictionary of metric scores
        hyperparams: Hyperparameters used
        seed: Random seed
        csv_path: Path to CSV file
    """
    # Ensure CSV exists with headers
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'experiment_id', 'experiment_name', 'model_name', 'features',
                'seed', 'accuracy', 'precision', 'recall', 'f1', 'auc',
                'hyperparams', 'timestamp'
            ])
    
    # Generate experiment ID
    experiment_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Write results
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            experiment_id,
            experiment_name,
            model_name,
            features,
            seed,
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1', 0),
            metrics.get('auc', 0),
            json.dumps(hyperparams),
            datetime.now().isoformat()
        ])
    
    console.print(f"[green]✓[/green] Experiment logged: {experiment_id}")

def display_metrics(metrics: Dict[str, float], title: str = "Model Performance"):
    """
    Display metrics in a nice table format
    
    Args:
        metrics: Dictionary of metric scores
        title: Table title
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Score", style="green")
    
    for metric, score in metrics.items():
        if isinstance(score, float):
            table.add_row(metric.capitalize(), f"{score:.4f}")
        else:
            table.add_row(metric.capitalize(), str(score))
    
    console.print(table)

def save_predictions(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    true_labels: Optional[np.ndarray],
    texts: List[str],
    output_path: Path
):
    """
    Save predictions to CSV file
    
    Args:
        predictions: Predicted labels
        probabilities: Prediction probabilities
        true_labels: True labels (if available)
        texts: Input texts
        output_path: Path to save predictions
    """
    df_data = {
        'text': texts,
        'predicted_label': predictions,
        'probability_fake': probabilities[:, 1] if probabilities.ndim > 1 else probabilities
    }
    
    if true_labels is not None:
        df_data['true_label'] = true_labels
        df_data['correct'] = predictions == true_labels
    
    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False)
    console.print(f"[green]✓[/green] Predictions saved to {output_path}")

def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Checkpoint dictionary
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    console.print(f"[green]✓[/green] Checkpoint loaded from {checkpoint_path}")
    return checkpoint

def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: Path
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_path: Path to save checkpoint
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model,
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)
    console.print(f"[green]✓[/green] Checkpoint saved to {checkpoint_path}")

def calculate_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        labels: Array of labels
    
    Returns:
        Dictionary mapping class to weight
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = {cls: total / (len(unique) * count) for cls, count in zip(unique, counts)}
    
    console.print("[yellow]Class distribution:[/yellow]")
    for cls, count in zip(unique, counts):
        console.print(f"  Class {cls}: {count} samples ({count/total*100:.1f}%), weight: {weights[cls]:.3f}")
    
    return weights

def clean_text(text: str) -> str:
    """
    Basic text cleaning
    
    Args:
        text: Input text
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove URLs
    import re
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove HTML tags if present
    text = re.sub(r'<.*?>', '', text)
    
    return text.strip()

def get_device() -> torch.device:
    """
    Get the best available device
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        console.print(f"[green]✓[/green] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        console.print("[yellow]⚠[/yellow] GPU not available, using CPU")
    
    return device

def print_model_info(model):
    """
    Print model architecture information
    
    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    console.print("\n[bold cyan]Model Information:[/bold cyan]")
    console.print(f"  Total parameters: {total_params:,}")
    console.print(f"  Trainable parameters: {trainable_params:,}")
    console.print(f"  Non-trainable parameters: {total_params - trainable_params:,}")

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like F1
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop training
        
        Args:
            score: Current score
        
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                console.print(f"[red]Early stopping triggered after {self.counter} epochs without improvement[/red]")
        
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if score improved"""
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta
