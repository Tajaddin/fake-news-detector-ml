"""
Baseline models for Fake News Detection
Implements Logistic Regression, SVM, and Naive Bayes classifiers
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import pickle
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import csr_matrix
from rich.console import Console
from rich.table import Table
import warnings
warnings.filterwarnings('ignore')

from config import BASELINE_CONFIG, MODELS_DIR, RANDOM_SEED
from utils import console, set_seed, display_metrics

class BaselineModel:
    """Base class for baseline models"""
    
    def __init__(
        self,
        model_type: str = 'logistic_regression',
        random_state: int = RANDOM_SEED,
        class_weight: str = 'balanced'
    ):
        """
        Initialize baseline model
        
        Args:
            model_type: Type of model to use
            random_state: Random seed
            class_weight: Class weight strategy
        """
        self.model_type = model_type
        self.random_state = random_state
        self.class_weight = class_weight
        self.model = None
        self.best_params_ = None
        self.cv_scores_ = None
        self.is_fitted = False
        
        set_seed(random_state)
        self.console = Console()
    
    def _create_model(self, **kwargs):
        """Create model based on type"""
        if self.model_type == 'logistic_regression':
            return LogisticRegression(
                solver='liblinear',
                max_iter=1000,
                random_state=self.random_state,
                class_weight=self.class_weight,
                **kwargs
            )
        
        elif self.model_type == 'svm':
            return LinearSVC(
                max_iter=1000,
                random_state=self.random_state,
                class_weight=self.class_weight,
                dual=False,
                **kwargs
            )
        
        elif self.model_type == 'naive_bayes':
            # Naive Bayes doesn't support class_weight directly
            return MultinomialNB(**kwargs)
        
        elif self.model_type == 'complement_nb':
            # Better for imbalanced datasets
            return ComplementNB(**kwargs)
        
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight=self.class_weight,
                n_jobs=-1,
                **kwargs
            )
        
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _get_param_grid(self) -> Dict:
        """Get hyperparameter grid for model type"""
        if self.model_type == 'logistic_regression':
            return {
                'C': BASELINE_CONFIG['logistic_regression']['C_values']
            }
        
        elif self.model_type == 'svm':
            return {
                'C': BASELINE_CONFIG['svm']['C_values']
            }
        
        elif self.model_type in ['naive_bayes', 'complement_nb']:
            return {
                'alpha': BASELINE_CONFIG['naive_bayes']['alpha_values']
            }
        
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        
        elif self.model_type == 'gradient_boosting':
            return {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        
        else:
            return {}
    
    def train(
        self,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: Optional[csr_matrix] = None,
        y_val: Optional[np.ndarray] = None,
        tune_hyperparams: bool = True,
        cv_folds: int = 5
    ):
        """
        Train the model with optional hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            tune_hyperparams: Whether to tune hyperparameters
            cv_folds: Number of cross-validation folds
        """
        self.console.print(f"\n[bold cyan]Training {self.model_type} model...[/bold cyan]")
        
        # Hyperparameter tuning
        if tune_hyperparams:
            param_grid = self._get_param_grid()
            
            if param_grid:
                self.console.print(f"  Performing grid search with {cv_folds}-fold CV...")
                
                # Create base model
                base_model = self._create_model()
                
                # Grid search
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=cv,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                self.model = grid_search.best_estimator_
                self.best_params_ = grid_search.best_params_
                self.cv_scores_ = {
                    'mean': grid_search.best_score_,
                    'std': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
                }
                
                self.console.print(f"  Best parameters: {self.best_params_}")
                self.console.print(f"  CV F1 score: {self.cv_scores_['mean']:.4f} "
                                 f"(+/- {self.cv_scores_['std']:.4f})")
            else:
                # No hyperparameters to tune
                self.model = self._create_model()
                self.model.fit(X_train, y_train)
        else:
            # Train with default parameters
            self.model = self._create_model()
            self.model.fit(X_train, y_train)
        
        # Calibrate probabilities for SVM (doesn't have predict_proba by default)
        if self.model_type == 'svm':
            self.console.print("  Calibrating SVM probabilities...")
            self.model = CalibratedClassifierCV(self.model, cv=3)
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.console.print(f"  Validation F1: {val_metrics['f1']:.4f}")
        
        self.console.print(f"[green]✓ {self.model_type} training complete[/green]")
    
    def predict(self, X: csr_matrix) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Feature matrix
        
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return binary predictions
            predictions = self.predict(X)
            proba = np.zeros((len(predictions), 2))
            proba[np.arange(len(predictions)), predictions] = 1.0
            return proba
    
    def evaluate(
        self,
        X: csr_matrix,
        y_true: np.ndarray,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y_true: True labels
            return_predictions: Whether to return predictions
        
        Returns:
            Dictionary of metrics (and predictions if requested)
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Add AUC if we have probabilities
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                metrics['auc'] = 0.0
        
        if return_predictions:
            return metrics, y_pred, y_proba
        
        return metrics
    
    def save(self, path: Optional[Path] = None):
        """
        Save model to disk
        
        Args:
            path: Path to save model (auto-generated if None)
        """
        if path is None:
            path = MODELS_DIR / f"{self.model_type}_model.pkl"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'best_params': self.best_params_,
            'cv_scores': self.cv_scores_,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, path)
        self.console.print(f"[green]✓ Model saved to {path}[/green]")
    
    @classmethod
    def load(cls, path: Path):
        """
        Load model from disk
        
        Args:
            path: Path to saved model
        
        Returns:
            BaselineModel instance
        """
        model_data = joblib.load(path)
        
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.best_params_ = model_data['best_params']
        instance.cv_scores_ = model_data['cv_scores']
        instance.is_fitted = model_data['is_fitted']
        
        return instance
    
    def get_feature_importance(self, feature_names: List[str], top_k: int = 20) -> pd.DataFrame:
        """
        Get feature importance for interpretable models
        
        Args:
            feature_names: List of feature names
            top_k: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        if self.model_type in ['logistic_regression', 'svm']:
            # Get coefficients for linear models
            if hasattr(self.model, 'coef_'):
                coef = self.model.coef_[0]
                
                # Get top positive and negative features
                importance_df = pd.DataFrame({
                    'feature': feature_names[:len(coef)],
                    'coefficient': coef
                })
                
                # Sort by absolute value
                importance_df['abs_coef'] = np.abs(importance_df['coefficient'])
                importance_df = importance_df.sort_values('abs_coef', ascending=False)
                
                return importance_df.head(top_k)
        
        elif self.model_type == 'random_forest':
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names[:len(self.model.feature_importances_)],
                    'importance': self.model.feature_importances_
                })
                
                return importance_df.sort_values('importance', ascending=False).head(top_k)
        
        return pd.DataFrame()


class ModelEnsemble:
    """Ensemble of multiple baseline models"""
    
    def __init__(self, models: List[BaselineModel], weights: Optional[List[float]] = None):
        """
        Initialize ensemble
        
        Args:
            models: List of baseline models
            weights: Weights for each model (uniform if None)
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.console = Console()
    
    def train(self, X_train: csr_matrix, y_train: np.ndarray,
             X_val: Optional[csr_matrix] = None, y_val: Optional[np.ndarray] = None):
        """Train all models in ensemble"""
        self.console.print("[bold cyan]Training ensemble models...[/bold cyan]")
        
        for model in self.models:
            model.train(X_train, y_train, X_val, y_val)
    
    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        """Weighted average of predictions"""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            proba = model.predict_proba(X)
            predictions.append(proba * weight)
        
        return np.sum(predictions, axis=0) / sum(self.weights)
    
    def predict(self, X: csr_matrix) -> np.ndarray:
        """Make ensemble predictions"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def evaluate(self, X: csr_matrix, y_true: np.ndarray) -> Dict[str, Any]:
        """Evaluate ensemble performance"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_proba[:, 1]) if len(np.unique(y_true)) == 2 else 0,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }


def train_all_baselines(
    X_train: csr_matrix,
    y_train: np.ndarray,
    X_val: csr_matrix,
    y_val: np.ndarray
) -> Dict[str, BaselineModel]:
    """
    Train all baseline models
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
    
    Returns:
        Dictionary of trained models
    """
    models = {}
    model_types = ['logistic_regression', 'svm', 'naive_bayes', 'complement_nb']
    
    for model_type in model_types:
        model = BaselineModel(model_type=model_type)
        model.train(X_train, y_train, X_val, y_val)
        models[model_type] = model
    
    return models


def compare_models(
    models: Dict[str, BaselineModel],
    X_test: csr_matrix,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Compare performance of multiple models
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for name, model in models.items():
        metrics = model.evaluate(X_test, y_test)
        
        results.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'AUC': metrics.get('auc', 0)
        })
    
    return pd.DataFrame(results).sort_values('F1', ascending=False)


def main():
    """Main training script for baseline models"""
    import argparse
    from data_io import DataLoader
    from features import create_baseline_features
    
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'logistic_regression', 'svm', 
                               'naive_bayes', 'complement_nb', 'ensemble'],
                       help='Model to train')
    parser.add_argument('--tune', action='store_true',
                       help='Tune hyperparameters')
    args = parser.parse_args()
    
    # Load data
    console.print("[cyan]Loading data...[/cyan]")
    loader = DataLoader()
    train_df, val_df, test_df = loader.load_processed_data()
    
    # Extract features
    console.print("[cyan]Extracting features...[/cyan]")
    feature_extractor = create_baseline_features()
    X_train = feature_extractor.fit_transform(train_df)
    X_val = feature_extractor.transform(val_df)
    X_test = feature_extractor.transform(test_df)
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    console.print(f"Feature shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # Train models
    if args.model == 'all':
        models = train_all_baselines(X_train, y_train, X_val, y_val)
        
        # Compare models
        console.print("\n[bold]Model Comparison on Test Set:[/bold]")
        comparison_df = compare_models(models, X_test, y_test)
        
        # Display results table
        table = Table(title="Baseline Model Performance", show_header=True, header_style="bold magenta")
        
        for col in comparison_df.columns:
            if col == 'Model':
                table.add_column(col, style="cyan", no_wrap=True)
            else:
                table.add_column(col, style="green")
        
        for _, row in comparison_df.iterrows():
            table.add_row(
                row['Model'],
                f"{row['Accuracy']:.4f}",
                f"{row['Precision']:.4f}",
                f"{row['Recall']:.4f}",
                f"{row['F1']:.4f}",
                f"{row['AUC']:.4f}"
            )
        
        console.print(table)
        
        # Save models
        for name, model in models.items():
            model.save()
    
    elif args.model == 'ensemble':
        # Train ensemble
        model_types = ['logistic_regression', 'svm', 'complement_nb']
        ensemble_models = [BaselineModel(model_type=mt) for mt in model_types]
        
        ensemble = ModelEnsemble(ensemble_models)
        ensemble.train(X_train, y_train, X_val, y_val)
        
        # Evaluate ensemble
        test_metrics = ensemble.evaluate(X_test, y_test)
        console.print("\n[bold]Ensemble Performance:[/bold]")
        display_metrics(test_metrics)
    
    else:
        # Train single model
        model = BaselineModel(model_type=args.model)
        model.train(X_train, y_train, X_val, y_val, tune_hyperparams=args.tune)
        
        # Evaluate
        test_metrics = model.evaluate(X_test, y_test)
        console.print(f"\n[bold]Test Set Performance ({args.model}):[/bold]")
        display_metrics(test_metrics)
        
        # Save model
        model.save()
    
    console.print("\n[bold green]✨ Training complete![/bold green]")


if __name__ == "__main__":
    main()
