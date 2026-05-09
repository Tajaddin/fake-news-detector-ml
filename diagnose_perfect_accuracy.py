"""
Comprehensive Diagnosis: Why is the model getting 100% accuracy?
This script checks all possible causes of unrealistic perfect classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def diagnose_perfect_accuracy():
    """Run comprehensive diagnosis"""
    
    print("="*70)
    print("DIAGNOSIS: WHY IS THE MODEL ACHIEVING 100% ACCURACY?")
    print("="*70)
    
    issues_found = []
    
    # 1. Check dataset size
    print("\n1. CHECKING DATASET SIZE...")
    print("-"*50)
    
    data_dir = Path("data/processed")
    train_df = pd.read_csv(data_dir / "train_processed_full.csv")
    val_df = pd.read_csv(data_dir / "val_processed_full.csv")
    test_df = pd.read_csv(data_dir / "test_processed_full.csv")
    
    total_samples = len(train_df) + len(val_df) + len(test_df)
    print(f"   Total samples: {total_samples}")
    print(f"   - Train: {len(train_df)}")
    print(f"   - Val: {len(val_df)}")
    print(f"   - Test: {len(test_df)}")
    
    if total_samples < 1000:
        issues_found.append("TINY DATASET: Only {} samples total".format(total_samples))
        print("   ⚠️ ISSUE: Dataset is too small for robust ML!")
    
    # 2. Check for data leakage
    print("\n2. CHECKING FOR DATA LEAKAGE...")
    print("-"*50)
    
    # Check text overlap
    train_texts = set(train_df['text'].values)
    test_texts = set(test_df['text'].values)
    overlap = train_texts.intersection(test_texts)
    
    if overlap:
        issues_found.append(f"DATA LEAKAGE: {len(overlap)} test samples appear in training!")
        print(f"   ⚠️ CRITICAL: {len(overlap)} test samples found in training set!")
        print("   Example leaked text:", list(overlap)[0][:100] + "...")
    else:
        print("   ✓ No direct text overlap between train and test")
    
    # 3. Check for artificially distinct patterns
    print("\n3. CHECKING FOR ARTIFICIAL PATTERNS...")
    print("-"*50)
    
    # Get texts by label
    train_fake = train_df[train_df['label'] == 1]['text'].str.lower()
    train_real = train_df[train_df['label'] == 0]['text'].str.lower()
    
    # Check for "magic words" that perfectly separate classes
    fake_words = set(' '.join(train_fake).split())
    real_words = set(' '.join(train_real).split())
    
    fake_only = fake_words - real_words
    real_only = real_words - fake_words
    
    print(f"   Words appearing ONLY in fake news: {len(fake_only)}")
    print(f"   Words appearing ONLY in real news: {len(real_only)}")
    
    if fake_only:
        print(f"   Example fake-only words: {list(fake_only)[:10]}")
    if real_only:
        print(f"   Example real-only words: {list(real_only)[:10]}")
    
    if len(fake_only) > 10 or len(real_only) > 10:
        issues_found.append(f"ARTIFICIAL PATTERNS: Classes have {len(fake_only)} and {len(real_only)} unique words")
        print("   ⚠️ ISSUE: Classes have completely distinct vocabularies!")
        print("   This suggests a synthetic/toy dataset.")
    
    # 4. Check for template-based generation
    print("\n4. CHECKING FOR TEMPLATE-BASED TEXT...")
    print("-"*50)
    
    # Check if texts follow patterns
    fake_starts = train_fake.str[:30].value_counts()
    real_starts = train_real.str[:30].value_counts()
    
    if fake_starts.iloc[0] > 5:
        issues_found.append(f"TEMPLATE TEXT: Fake news has repeated patterns")
        print(f"   ⚠️ Fake news texts repeat this start {fake_starts.iloc[0]} times:")
        print(f"      '{fake_starts.index[0]}'")
    
    if real_starts.iloc[0] > 5:
        issues_found.append(f"TEMPLATE TEXT: Real news has repeated patterns")
        print(f"   ⚠️ Real news texts repeat this start {real_starts.iloc[0]} times:")
        print(f"      '{real_starts.index[0]}'")
    
    if fake_starts.iloc[0] <= 5 and real_starts.iloc[0] <= 5:
        print("   ✓ No obvious template patterns detected")
    
    # 5. Test with random permutation
    print("\n5. TESTING WITH RANDOM LABELS...")
    print("-"*50)
    
    # Shuffle labels randomly
    y_train_random = np.random.randint(0, 2, size=len(train_df))
    y_test_random = np.random.randint(0, 2, size=len(test_df))
    
    # Train simple model with random labels
    vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.95)
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train_random)
    
    random_acc = model.score(X_test, y_test_random)
    print(f"   Accuracy with random labels: {random_acc:.3f}")
    
    if random_acc < 0.6:
        print("   ✓ Random labels give poor accuracy (as expected)")
    else:
        print("   ⚠️ Even random labels give decent accuracy!")
        issues_found.append(f"OVERFITTING: Random labels achieve {random_acc:.1%} accuracy")
    
    # 6. Train with real labels for comparison
    print("\n6. TESTING WITH REAL LABELS...")
    print("-"*50)
    
    y_train_real = train_df['label'].values
    y_test_real = test_df['label'].values
    
    model_real = LogisticRegression(random_state=42, C=0.01)
    model_real.fit(X_train, y_train_real)
    
    real_acc = model_real.score(X_test, y_test_real)
    train_acc = model_real.score(X_train, y_train_real)
    
    print(f"   Training accuracy: {train_acc:.3f}")
    print(f"   Test accuracy: {real_acc:.3f}")
    
    if real_acc >= 0.99:
        issues_found.append(f"PERFECT CLASSIFICATION: {real_acc:.1%} test accuracy")
        print("   ⚠️ Near-perfect accuracy achieved!")
        
        # Check which features are most important
        coef = model_real.coef_[0]
        feature_names = vectorizer.get_feature_names_out()
        top_fake_idx = coef.argsort()[-5:][::-1]
        top_real_idx = coef.argsort()[:5]
        
        print("\n   Top features for FAKE news:")
        for idx in top_fake_idx:
            print(f"      {feature_names[idx]:20s} (coef={coef[idx]:.3f})")
        
        print("\n   Top features for REAL news:")
        for idx in top_real_idx:
            print(f"      {feature_names[idx]:20s} (coef={coef[idx]:.3f})")
    
    # 7. Check actual text examples
    print("\n7. EXAMINING ACTUAL TEXT SAMPLES...")
    print("-"*50)
    
    print("\nFake news example:")
    print("   " + train_fake.iloc[0][:200] + "...")
    
    print("\nReal news example:")
    print("   " + train_real.iloc[0][:200] + "...")
    
    # Check if certain keywords always appear
    fake_keywords = ['fake', 'hoax', 'aliens', 'miracle', 'conspiracy']
    real_keywords = ['report', 'official', 'government', 'announced', 'according']
    
    for keyword in fake_keywords:
        pct = (train_fake.str.contains(keyword).sum() / len(train_fake)) * 100
        if pct > 50:
            issues_found.append(f"KEYWORD PATTERN: '{keyword}' in {pct:.0f}% of fake news")
            print(f"   ⚠️ '{keyword}' appears in {pct:.0f}% of fake news!")
    
    for keyword in real_keywords:
        pct = (train_real.str.contains(keyword).sum() / len(train_real)) * 100
        if pct > 50:
            issues_found.append(f"KEYWORD PATTERN: '{keyword}' in {pct:.0f}% of real news")
            print(f"   ⚠️ '{keyword}' appears in {pct:.0f}% of real news!")
    
    # 8. Final diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    
    if issues_found:
        print("\n⚠️ ISSUES FOUND:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        
        print("\n📊 CONCLUSION:")
        if "DATA LEAKAGE" in str(issues_found):
            print("   Your test set is contaminated with training data!")
            print("   Fix: Ensure proper train/test split with no overlap.")
        elif "ARTIFICIAL PATTERNS" in str(issues_found) or "TEMPLATE TEXT" in str(issues_found):
            print("   You're using a TOY/SYNTHETIC dataset with artificial patterns!")
            print("   Fix: Use a real-world dataset like:")
            print("   - LIAR dataset")
            print("   - FakeNewsNet")
            print("   - ISOT Fake News dataset")
            print("   - Getting Real about Fake News dataset")
        elif "TINY DATASET" in str(issues_found):
            print("   Your dataset is too small for meaningful ML!")
            print("   Fix: Get at least 5,000-10,000 samples.")
        elif "KEYWORD PATTERN" in str(issues_found):
            print("   Your data has obvious keyword patterns that make classification trivial!")
            print("   Fix: Use more realistic, diverse data.")
        
        print("\n💡 RECOMMENDATIONS:")
        print("   1. Use a standard fake news dataset from research papers")
        print("   2. Ensure proper train/val/test split (60/20/20 or 80/10/10)")
        print("   3. Never fit TF-IDF on test data")
        print("   4. Expect realistic accuracy: 60-85% for fake news detection")
    else:
        print("\n✓ No obvious issues found.")
        print("  If still getting 100% accuracy, check:")
        print("  - Your preprocessing pipeline")
        print("  - The original data source")
        print("  - Feature extraction code")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    diagnose_perfect_accuracy()
