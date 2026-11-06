"""
Comprehensive Diagnosis: Why is the model getting 100% accuracy?
This script checks all possible causes of unrealistic perfect classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')


def shuffled_label_baseline(model, X_train, y_train, X_test, y_test, n_permutations=50, seed=42):
    """Test accuracies of the model refit on permuted training labels.

    A permutation keeps the class balance but cuts every link between text and
    label, so these scores are what the model reaches on this split with no
    signal; they center on the majority-class rate. The real-label score means
    something only if it clears them. This is a null baseline, not a leak
    detector: permuting the labels hides a leak as well.
    """
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train)
    return np.array([clone(model).fit(X_train, rng.permutation(y_train)).score(X_test, y_test)
                     for _ in range(n_permutations)])


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
    
    # 5. Null baseline: the same model refit on permuted training labels
    print("\n5. BUILDING A SHUFFLED-LABEL BASELINE...")
    print("-"*50)

    vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.95)
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])

    y_train_real = train_df['label'].values
    y_test_real = test_df['label'].values
    model = LogisticRegression(random_state=42, C=0.01)

    null_scores = shuffled_label_baseline(model, X_train, y_train_real, X_test, y_test_real)
    null_p95 = np.percentile(null_scores, 95)
    majority_acc = max(y_test_real.mean(), 1 - y_test_real.mean())
    print(f"   Test accuracy over {len(null_scores)} label permutations: mean {null_scores.mean():.3f}, "
          f"95th percentile {null_p95:.3f} (majority class: {majority_acc:.3f})")

    # 6. Train with real labels for comparison
    print("\n6. TESTING WITH REAL LABELS...")
    print("-"*50)
    
    model_real = clone(model).fit(X_train, y_train_real)
    
    real_acc = model_real.score(X_test, y_test_real)
    train_acc = model_real.score(X_train, y_train_real)
    
    print(f"   Training accuracy: {train_acc:.3f}")
    print(f"   Test accuracy: {real_acc:.3f}")
    
    if real_acc <= null_p95:
        issues_found.append(f"NO SIGNAL: {real_acc:.1%} test accuracy does not beat the shuffled-label "
                            f"baseline's 95th percentile ({null_p95:.1%})")
        print("   ⚠️ No better than the same model trained on shuffled labels!")
    else:
        print(f"   ✓ Beats the shuffled-label baseline's 95th percentile ({null_p95:.3f})")
    
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
        elif "NO SIGNAL" in str(issues_found):
            print("   The model does no better than it does with shuffled labels!")
            print("   Fix: Check that the labels match the texts and that the features carry signal.")
        
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
    # Windows consoles default to a legacy code page; the output uses symbols like ✓.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    diagnose_perfect_accuracy()
