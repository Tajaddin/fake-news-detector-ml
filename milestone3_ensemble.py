"""
Ensemble Model for Fake News Detection
Combines baseline ML models with transformer predictions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
import pickle
import joblib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FakeNewsEnsemble:
    """
    Ensemble model combining multiple approaches:
    1. Traditional ML models (Logistic Regression, SVM, Naive Bayes)
    2. Transformer model (DistilBERT)
    3. Metadata features
    4. Multiple ensemble strategies (voting, stacking, weighted average)
    """
    
    def __init__(self):
        self.base_models = {}
        self.transformer_model = None
        self.meta_model = None
        self.ensemble_strategy = 'weighted_average'
        self.model_weights = None
        self.transformer_probabilities = None
        
    def add_base_model(self, name, model, weight=1.0):
        """Add a base model to the ensemble"""
        self.base_models[name] = {
            'model': model,
            'weight': weight,
            'performance': {}
        }
    
    def add_transformer_predictions(self, predictions, probabilities):
        """Add transformer model predictions"""
        self.transformer_predictions = predictions
        self.transformer_probabilities = probabilities
    
    def train_stacking_ensemble(self, X_train, y_train, X_val, y_val):
        """Train a stacking ensemble with meta-learner"""
        print("   -> Generating meta-features for stacking...")
        
        # Collect base model predictions
        train_meta_features = []
        val_meta_features = []
        
        for name, model_dict in self.base_models.items():
            model = model_dict['model']
            
            # Get predictions on training set
            if hasattr(model, 'predict_proba'):
                train_pred = model.predict_proba(X_train)[:, 1]
                val_pred = model.predict_proba(X_val)[:, 1]
            else:
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
            
            train_meta_features.append(train_pred)
            val_meta_features.append(val_pred)
        
        # Add transformer predictions if available
        if self.transformer_probabilities is not None:
            # IMPORTANT: For stacking training, transformer_probabilities must contain
            # [Train_Probs, Val_Probs] concatenated in that order.
            train_size = X_train.shape[0]  # FIXED: Use shape[0] instead of len()
            
            # Safety check
            if len(self.transformer_probabilities) < train_size + X_val.shape[0]:
                print(f"Warning: Transformer probs length ({len(self.transformer_probabilities)}) "
                      f"is smaller than Train+Val size ({train_size + X_val.shape[0]}). Stacking might fail.")
            
            train_meta_features.append(self.transformer_probabilities[:train_size])
            val_meta_features.append(self.transformer_probabilities[train_size:train_size+X_val.shape[0]])
        
        # Stack features
        X_meta_train = np.column_stack(train_meta_features)
        X_meta_val = np.column_stack(val_meta_features)
        
        # Train meta-learner
        self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
        self.meta_model.fit(X_meta_train, y_train)
        
        # Evaluate on validation
        meta_pred = self.meta_model.predict(X_meta_val)
        meta_prob = self.meta_model.predict_proba(X_meta_val)[:, 1]
        
        accuracy = accuracy_score(y_val, meta_pred)
        f1 = f1_score(y_val, meta_pred, average='weighted')
        auc = roc_auc_score(y_val, meta_prob)
        
        print(f"   -> Stacking Validation Accuracy: {accuracy:.4f}")
        print(f"   -> Stacking Validation F1: {f1:.4f}")
        
        return self.meta_model
    
    def weighted_average_ensemble(self, predictions_dict, weights=None):
        """Combine predictions using weighted average"""
        if weights is None:
            weights = {name: 1.0 for name in predictions_dict.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate weighted average
        weighted_sum = None
        for name, preds in predictions_dict.items():
            weight = normalized_weights.get(name, 0)
            if weighted_sum is None:
                weighted_sum = weight * preds
            else:
                weighted_sum += weight * preds
        
        return weighted_sum
    
    def majority_voting_ensemble(self, predictions_dict):
        """Combine predictions using majority voting"""
        predictions_array = np.array(list(predictions_dict.values()))
        # Use mode for each sample
        ensemble_predictions = stats.mode(predictions_array, axis=0)[0].flatten()
        return ensemble_predictions
    
    def predict_ensemble(self, X, strategy='weighted_average'):
        """Make ensemble predictions using specified strategy"""
        predictions_dict = {}
        probabilities_dict = {}
        
        # Get predictions from all base models
        for name, model_dict in self.base_models.items():
            model = model_dict['model']
            
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[:, 1]
                preds = (probs > 0.5).astype(int)
                probabilities_dict[name] = probs
            else:
                preds = model.predict(X)
                probabilities_dict[name] = preds
            
            predictions_dict[name] = preds
        
        # Add transformer predictions if available
        # NOTE: self.transformer_probabilities must match X length
        if self.transformer_probabilities is not None:
            # Ensure lengths match to avoid shape mismatch errors
            if len(self.transformer_probabilities) == X.shape[0]:
                predictions_dict['transformer'] = (self.transformer_probabilities > 0.5).astype(int)
                probabilities_dict['transformer'] = self.transformer_probabilities
            else:
                # Fallback if dimensions don't match (should be handled by caller)
                pass 
        
        # Apply ensemble strategy
        if strategy == 'weighted_average':
            ensemble_probs = self.weighted_average_ensemble(
                probabilities_dict, 
                self.model_weights
            )
            ensemble_preds = (ensemble_probs > 0.5).astype(int)
        
        elif strategy == 'majority_voting':
            ensemble_preds = self.majority_voting_ensemble(predictions_dict)
            # Probability for voting is just average of binary preds or similar
            ensemble_probs = self.weighted_average_ensemble(probabilities_dict)
        
        elif strategy == 'stacking':
            if self.meta_model is None:
                raise ValueError("Meta-model not trained. Call train_stacking_ensemble first.")
            
            # Ensure meta features are in same order as training
            meta_features = np.column_stack(list(probabilities_dict.values()))
            ensemble_preds = self.meta_model.predict(meta_features)
            ensemble_probs = self.meta_model.predict_proba(meta_features)[:, 1]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return ensemble_preds, ensemble_probs
    
    def optimize_weights(self, X_val, y_val):
        """Optimize model weights based on validation performance"""
        best_weights = None
        best_f1 = 0
        
        # Simple grid search 
        weight_options = [0.5, 1.0, 1.5, 2.0]
        
        for w1 in weight_options: # Logistic
            for w2 in weight_options: # SVM
                for w3 in weight_options: # NB
                    weights = {
                        'logistic': w1,
                        'svm': w2,
                        'naive_bayes': w3,
                        'transformer': 2.0 
                    }
                    
                    self.model_weights = weights
                    try:
                        preds, _ = self.predict_ensemble(X_val, strategy='weighted_average')
                        f1 = f1_score(y_val, preds, average='weighted')
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            best_weights = weights.copy()
                    except Exception:
                        continue
        
        self.model_weights = best_weights
        print(f"\nOptimized weights: {best_weights}")
        print(f"Best validation F1: {best_f1:.4f}")
        return best_weights
    
    def evaluate_all_strategies(self, X_test, y_test):
        """Evaluate all ensemble strategies"""
        results = {}
        strategies = ['weighted_average', 'majority_voting', 'stacking']
        
        for strategy in strategies:
            try:
                preds, probs = self.predict_ensemble(X_test, strategy=strategy)
                
                accuracy = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average='weighted')
                if len(np.unique(probs)) > 1:
                    auc = roc_auc_score(y_test, probs)
                else:
                    auc = 0.5
                
                results[strategy] = {
                    'accuracy': accuracy, 'f1': f1, 'auc': auc
                }
                
                print(f"\n{strategy.upper()} Strategy:")
                print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
                
            except Exception as e:
                print(f"Skipping {strategy}: {str(e)}")
                results[strategy] = None
        
        return results
