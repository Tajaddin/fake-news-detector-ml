"""
Test feature extraction and model prediction manually
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('src')

from features import create_baseline_features, FeatureExtractor
from models_baseline import BaselineModel
import joblib

def test_feature_extraction():
    """Test feature extraction behavior"""
    
    print("="*60)
    print("TESTING FEATURE EXTRACTION")
    print("="*60)
    
    # Load the trained feature extractor
    feature_path = Path("models/feature_extractor_logistic_regression.pkl")
    if not feature_path.exists():
        print(f"Feature extractor not found at {feature_path}")
        return
    
    # Load model too
    model_path = Path("models/logistic_regression_42.pkl") 
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    
    # Load both
    feature_extractor = FeatureExtractor.load(feature_path)
    model_data = joblib.load(model_path)
    model = model_data['model']
    
    print("\n1. FEATURE EXTRACTOR INFO:")
    print(f"   Max features: {feature_extractor.max_features}")
    print(f"   N-gram range: {feature_extractor.ngram_range}")
    print(f"   Min DF: {feature_extractor.min_df}")
    print(f"   Max DF: {feature_extractor.max_df}")
    
    # Check vocabulary size
    vocab_size = len(feature_extractor.text_vectorizer.vocabulary_)
    print(f"   Actual vocabulary size: {vocab_size}")
    
    # Get feature names
    feature_names = feature_extractor.get_feature_names()
    print(f"   Total features: {len(feature_names)}")
    
    # Show some feature examples
    print("\n2. SAMPLE FEATURES:")
    print("   First 10 features:")
    for i, name in enumerate(feature_names[:10]):
        print(f"     {i}: {name}")
    
    # Test on custom examples
    print("\n3. TESTING ON CUSTOM EXAMPLES:")
    
    test_examples = [
        # Should be classified as fake
        "BREAKING: Scientists discover miracle cure that doctors hate! Aliens confirmed to be living among us!",
        "Shocking discovery: Government admits to hiding alien technology for decades!",
        
        # Should be classified as real  
        "The city council approved a new infrastructure budget for road improvements this year.",
        "Federal Reserve announces interest rate decision following economic review.",
        
        # Ambiguous
        "New study shows surprising results about common household items.",
        "Local residents report unusual activity in the area last night."
    ]
    
    # Create a DataFrame
    test_df = pd.DataFrame({'text': test_examples})
    
    # Extract features
    X_test = feature_extractor.transform(test_df)
    
    print(f"\n   Feature matrix shape: {X_test.shape}")
    print(f"   Sparsity: {1 - X_test.nnz / (X_test.shape[0] * X_test.shape[1]):.2%}")
    
    # Predict
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    print("\n   Predictions:")
    for i, (text, pred, prob) in enumerate(zip(test_examples, predictions, probabilities[:, 1])):
        label = "FAKE" if pred == 1 else "REAL"
        print(f"\n   Example {i+1}: {label} (confidence: {prob:.3f})")
        print(f"   Text: {text[:100]}...")
    
    # Check which features activate for each example
    print("\n4. FEATURE ACTIVATION ANALYSIS:")
    
    for i, text in enumerate(test_examples[:2]):  # Check first 2 examples
        features_vec = X_test[i].toarray().flatten()
        active_indices = np.where(features_vec > 0)[0]
        
        print(f"\n   Example {i+1}: '{text[:50]}...'")
        print(f"   Active features: {len(active_indices)}/{len(features_vec)}")
        
        if len(active_indices) > 0:
            # Get top 5 features by TF-IDF value
            top_indices = active_indices[np.argsort(features_vec[active_indices])[-5:]]
            print("   Top features:")
            for idx in reversed(top_indices):
                if idx < len(feature_names):
                    print(f"     - {feature_names[idx]}: {features_vec[idx]:.3f}")
    
    # Load actual test data and check
    print("\n5. CHECKING ACTUAL TEST DATA:")
    
    test_df = pd.read_csv("data/processed/test_processed.csv")
    X_real_test = feature_extractor.transform(test_df)
    y_test = test_df['label'].values
    
    predictions = model.predict(X_real_test)
    accuracy = (predictions == y_test).mean()
    
    print(f"   Test set accuracy: {accuracy:.4f}")
    
    # Check a few misclassified examples (if any)
    misclassified = np.where(predictions != y_test)[0]
    if len(misclassified) > 0:
        print(f"   Found {len(misclassified)} misclassified examples")
        for idx in misclassified[:3]:
            print(f"\n   Misclassified example {idx}:")
            print(f"   Text: {test_df.iloc[idx]['text'][:150]}...")
            print(f"   True label: {y_test[idx]}, Predicted: {predictions[idx]}")
    else:
        print("   ⚠️ NO MISCLASSIFIED EXAMPLES - Perfect classification!")
        print("   This is highly suspicious for a real-world dataset.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_feature_extraction()
