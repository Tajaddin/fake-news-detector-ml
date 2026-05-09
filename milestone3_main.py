"""
Main Script for Milestone III: Fake News Detection (Robust & Fast Version)
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader as TorchDataLoader

# Add 'src' to path so config.py can be found
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / 'src'))

try:
    from config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, EXPERIMENTS_DIR
    from data_io import DataLoader
except ImportError:
    print("WARNING: Could not import config/data_io. Using defaults.")
    PROCESSED_DATA_DIR = Path("data/processed")
    MODELS_DIR = Path("models")
    FIGURES_DIR = Path("reports/figs")
    EXPERIMENTS_DIR = Path("experiments")
    class DataLoader:
        def load_processed_data(self):
            return (pd.read_csv(PROCESSED_DATA_DIR/"train_processed.csv"),
                    pd.read_csv(PROCESSED_DATA_DIR/"val_processed.csv"),
                    pd.read_csv(PROCESSED_DATA_DIR/"test_processed.csv"))

from milestone3_transformer_model import FakeNewsTransformer, FakeNewsDataset
from milestone3_ensemble import FakeNewsEnsemble
from milestone3_explainability import FakeNewsExplainer

# Ensure directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("="*80)
    print(" MILESTONE III: FULL EXECUTION (OPTIMIZED)")
    print("="*80)
    
    # ---------------------------------------------------------
    # 1. Data Loading
    # ---------------------------------------------------------
    print("\n[1/6] Loading Dataset...")
    try:
        loader = DataLoader()
        train_df, val_df, test_df = loader.load_processed_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    def prepare_text(df):
        if 'title' in df.columns and 'text' in df.columns:
            return (df['title'].fillna('') + ' ' + df['text'].fillna('')).tolist()
        return df['text'].fillna('').tolist()

    train_texts = prepare_text(train_df)
    val_texts = prepare_text(val_df)
    test_texts = prepare_text(test_df)
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # GPU Check & Slicing
    if not torch.cuda.is_available():
        print("\n⚠️  CPU DETECTED: Slicing data for faster completion...")
        LIMIT_TRAIN = 500
        LIMIT_EVAL = 200
        train_texts = train_texts[:LIMIT_TRAIN]
        y_train = y_train[:LIMIT_TRAIN]
        val_texts = val_texts[:LIMIT_EVAL]
        y_val = y_val[:LIMIT_EVAL]
        test_texts = test_texts[:LIMIT_EVAL]
        y_test = y_test[:LIMIT_EVAL]
    else:
        print("\n✅ GPU DETECTED: Running on full dataset.")

    print(f"Final Counts: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    # ---------------------------------------------------------
    # 2. Transformer Model (DistilBERT)
    # ---------------------------------------------------------
    print("\n[2/6] Training Transformer Model...")
    transformer = FakeNewsTransformer(model_name='distilbert-base-uncased')
    
    batch_size = 16 if torch.cuda.is_available() else 8
    
    # Create DataLoaders
    train_loader = TorchDataLoader(FakeNewsDataset(train_texts, y_train, transformer.tokenizer), batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(FakeNewsDataset(val_texts, y_val, transformer.tokenizer), batch_size=batch_size)
    test_loader = TorchDataLoader(FakeNewsDataset(test_texts, y_test, transformer.tokenizer), batch_size=batch_size)

    # Train
    print("Starting training loop (Epochs=1 for speed)...")
    transformer.train(train_loader, val_loader, epochs=1, learning_rate=2e-5)
    
    # Generate Predictions for ALL sets (Needed for Stacking)
    print("Generating transformer predictions for Ensemble...")
    _, _, _, (_, _, train_probs) = transformer.evaluate(train_loader) # Critical for stacking
    _, _, _, (_, _, val_probs) = transformer.evaluate(val_loader)
    _, _, _, (_, _, test_probs) = transformer.evaluate(test_loader)

    # ---------------------------------------------------------
    # 3. Vectorize Data (TF-IDF)
    # ---------------------------------------------------------
    print("\n[3/6] Vectorizing Text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_val_tfidf = vectorizer.transform(val_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    # ---------------------------------------------------------
    # 4. Ensemble Training
    # ---------------------------------------------------------
    print("\n[4/6] Building Ensemble...")
    ensemble = FakeNewsEnsemble()
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.naive_bayes import MultinomialNB
    
    # Train Base Models
    print("Training base models...")
    lr = LogisticRegression(class_weight='balanced', max_iter=500)
    lr.fit(X_train_tfidf, y_train)
    ensemble.add_base_model('logistic', lr)
    
    svm = LinearSVC(class_weight='balanced', dual=False)
    svm_cal = CalibratedClassifierCV(svm)
    svm_cal.fit(X_train_tfidf, y_train)
    ensemble.add_base_model('svm', svm_cal)
    
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    ensemble.add_base_model('naive_bayes', nb)

    # A. Optimize Weights (Needs Val Probs)
    print("Optimizing ensemble weights...")
    ensemble.transformer_probabilities = np.array(val_probs)
    ensemble.optimize_weights(X_val_tfidf, y_val)
    
    # B. Train Stacking (Needs Train + Val Probs Concatenated)
    print("Training Stacking Meta-Learner...")
    combined_probs = np.concatenate([train_probs, val_probs])
    ensemble.transformer_probabilities = combined_probs
    # This call slices the combined_probs internally based on X_train size
    ensemble.train_stacking_ensemble(X_train_tfidf, y_train, X_val_tfidf, y_val)

    # C. Final Evaluation (Needs Test Probs)
    print("Evaluating Ensemble Strategies...")
    ensemble.transformer_probabilities = np.array(test_probs)
    ensemble_results = ensemble.evaluate_all_strategies(X_test_tfidf, y_test)

    # ---------------------------------------------------------
    # 5. Explainability
    # ---------------------------------------------------------
    print("\n[5/6] Explainability Analysis...")
    try:
        explainer = FakeNewsExplainer(model_type='transformer', model=transformer.model)
        explainer.init_lime()
        
        idx = 0 
        sample_text = test_texts[idx]
        print(f"Explaining test sample #{idx}...")
        
        report = explainer.create_explanation_report(sample_text)
        fig = explainer.visualize_explanation(report, save_path=FIGURES_DIR / 'milestone3_explanation.png')
        print(f"Explanation saved to {FIGURES_DIR / 'milestone3_explanation.png'}")
        
    except Exception as e:
        print(f"Warning: Explainability step had an issue: {e}")
        report = {}

    # ---------------------------------------------------------
    # 6. Save & Finish
    # ---------------------------------------------------------
    print("\n[6/6] Saving Results...")
    results_path = EXPERIMENTS_DIR / 'milestone3_final_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump({'ensemble': ensemble_results, 'report': report}, f)

    print("\n" + "="*80)
    print(" SUCCESS! MILESTONE 3 COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()
