"""
Configuration file for Fake News Detection project
Contains all project-wide constants, paths, and hyperparameters
"""

from pathlib import Path
from typing import Dict, Any
import torch

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Experiment tracking
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUNS_CSV = EXPERIMENTS_DIR / "runs.csv"
MLFLOW_DIR = EXPERIMENTS_DIR / "mlruns"

# Reports and figures
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figs"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Baseline model configurations
BASELINE_CONFIG = {
    "tfidf": {
        "ngram_range": (1, 2),
        "min_df": 3,
        "max_df": 0.9,
        "max_features": 10000,
        "use_idf": True,
        "sublinear_tf": True
    },
    "logistic_regression": {
        "solver": "liblinear",
        "max_iter": 1000,
        "C_values": [0.01, 0.1, 1.0, 10.0],
        "class_weight": "balanced"
    },
    "svm": {
        "kernel": "linear",
        "C_values": [0.01, 0.1, 1.0, 10.0],
        "class_weight": "balanced"
    },
    "naive_bayes": {
        "alpha_values": [0.01, 0.1, 0.5, 1.0]
    }
}

# Transformer configurations
TRANSFORMER_CONFIG = {
    "model_name": "distilbert-base-uncased",  # Can switch to "bert-base-uncased"
    "max_length": 512,
    "batch_size": 16,
    "num_epochs": 3,
    "learning_rate": 2e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_clip_val": 1.0,
    "num_labels": 2,
    "dropout": 0.1
}

# Training settings
TRAINING_CONFIG = {
    "early_stopping_patience": 3,
    "save_best_only": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True,
    "num_seeds": 3,  # For multiple runs
    "validation_freq": 1  # Validate every N epochs
}

# Explainability settings
EXPLAIN_CONFIG = {
    "lime": {
        "num_features": 10,
        "num_samples": 5000
    },
    "shap": {
        "max_display": 20,
        "sample_size": 100
    }
}

# Threshold search settings
THRESHOLD_CONFIG = {
    "search_range": (0.1, 0.9),
    "num_thresholds": 50,
    "optimization_metric": "f1"  # Can be "f1", "precision", "recall", or "custom_loss"
}

# Label mapping
LABEL_MAP = {
    "real": 0,
    "fake": 1,
    "true": 0,
    "false": 1,
    "reliable": 0,
    "unreliable": 1
}

# Inverse label map for predictions
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items() if k in ["real", "fake"]}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S"
}

def get_config() -> Dict[str, Any]:
    """Return complete configuration dictionary"""
    return {
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "data": str(DATA_DIR),
            "models": str(MODELS_DIR),
            "experiments": str(EXPERIMENTS_DIR),
            "reports": str(REPORTS_DIR)
        },
        "seed": RANDOM_SEED,
        "device": str(DEVICE),
        "baseline": BASELINE_CONFIG,
        "transformer": TRANSFORMER_CONFIG,
        "training": TRAINING_CONFIG,
        "explain": EXPLAIN_CONFIG,
        "threshold": THRESHOLD_CONFIG
    }
