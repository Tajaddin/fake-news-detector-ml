"""
Investigate the fake news dataset for potential issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import hashlib

def investigate_data():
    """Comprehensive data investigation"""
    
    # Load the processed data
    data_dir = Path("data/processed")
    
    print("="*60)
    print("FAKE NEWS DATASET INVESTIGATION")
    print("="*60)
    
    # Load all splits
    train_df = pd.read_csv(data_dir / "train_processed_full.csv")
    val_df = pd.read_csv(data_dir / "val_processed_full.csv")
    test_df = pd.read_csv(data_dir / "test_processed_full.csv")
    
    # 1. Check dataset sizes
    print("\n1. DATASET SIZES:")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")
    print(f"   Test:  {len(test_df)} samples")
    print(f"   Total: {len(train_df) + len(val_df) + len(test_df)} samples")
    
    # 2. Check for duplicates
    print("\n2. DUPLICATE ANALYSIS:")
    
    # Check for duplicate texts within each set
    train_dups = train_df['text'].duplicated().sum()
    val_dups = val_df['text'].duplicated().sum()
    test_dups = test_df['text'].duplicated().sum()
    
    print(f"   Duplicates in train: {train_dups}")
    print(f"   Duplicates in val:   {val_dups}")
    print(f"   Duplicates in test:  {test_dups}")
    
    # Check for overlapping samples between sets
    train_texts = set(train_df['text'].values)
    val_texts = set(val_df['text'].values)
    test_texts = set(test_df['text'].values)
    
    train_val_overlap = len(train_texts.intersection(val_texts))
    train_test_overlap = len(train_texts.intersection(test_texts))
    val_test_overlap = len(val_texts.intersection(test_texts))
    
    print(f"\n   CROSS-SET OVERLAPS (DATA LEAKAGE CHECK):")
    print(f"   Train-Val overlap:  {train_val_overlap} samples")
    print(f"   Train-Test overlap: {train_test_overlap} samples")
    print(f"   Val-Test overlap:   {val_test_overlap} samples")
    
    if train_test_overlap > 0:
        print("   ⚠️ WARNING: DATA LEAKAGE DETECTED! Test samples appear in training!")
    
    # 3. Text length analysis
    print("\n3. TEXT LENGTH STATISTICS:")
    
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        lengths = df['text'].str.len()
        print(f"\n   {name}:")
        print(f"   - Mean length:   {lengths.mean():.1f} chars")
        print(f"   - Median length: {lengths.median():.1f} chars")
        print(f"   - Min length:    {lengths.min()} chars")
        print(f"   - Max length:    {lengths.max()} chars")
    
    # 4. Sample some texts to see patterns
    print("\n4. SAMPLE TEXTS BY LABEL:")
    
    print("\n   FAKE NEWS SAMPLES (Train):")
    fake_samples = train_df[train_df['label'] == 1]['text'].head(3)
    for i, text in enumerate(fake_samples, 1):
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"   {i}. {preview}")
    
    print("\n   REAL NEWS SAMPLES (Train):")
    real_samples = train_df[train_df['label'] == 0]['text'].head(3)
    for i, text in enumerate(real_samples, 1):
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"   {i}. {preview}")
    
    # 5. Check for obvious patterns in text
    print("\n5. OBVIOUS PATTERN DETECTION:")
    
    # Words that appear ONLY in fake news
    fake_texts = train_df[train_df['label'] == 1]['text'].str.lower().str.cat(sep=' ')
    real_texts = train_df[train_df['label'] == 0]['text'].str.lower().str.cat(sep=' ')
    
    fake_words = set(fake_texts.split())
    real_words = set(real_texts.split())
    
    fake_only_words = fake_words - real_words
    real_only_words = real_words - fake_words
    
    print(f"\n   Words appearing ONLY in fake news: {len(fake_only_words)}")
    if fake_only_words:
        print(f"   Examples: {list(fake_only_words)[:10]}")
    
    print(f"\n   Words appearing ONLY in real news: {len(real_only_words)}")
    if real_only_words:
        print(f"   Examples: {list(real_only_words)[:10]}")
    
    # 6. Check if labels are encoded in text
    print("\n6. LABEL LEAKAGE CHECK:")
    
    suspicious_patterns = ['fake', 'real', 'true', 'false', 'hoax', 'fact']
    
    for pattern in suspicious_patterns:
        fake_count = train_df[train_df['label'] == 1]['text'].str.lower().str.contains(pattern).sum()
        real_count = train_df[train_df['label'] == 0]['text'].str.lower().str.contains(pattern).sum()
        
        if fake_count > 0 or real_count > 0:
            print(f"   '{pattern}': Fake={fake_count}, Real={real_count}")
    
    # 7. Vocabulary size analysis
    print("\n7. VOCABULARY ANALYSIS:")
    
    all_texts = pd.concat([train_df['text'], val_df['text'], test_df['text']])
    all_words = ' '.join(all_texts.str.lower()).split()
    vocab = set(all_words)
    word_freq = Counter(all_words)
    
    print(f"   Total unique words: {len(vocab)}")
    print(f"   Total word count: {len(all_words)}")
    print(f"   Most common words: {word_freq.most_common(10)}")
    
    # 8. Check class balance
    print("\n8. CLASS BALANCE:")
    
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        fake_count = (df['label'] == 1).sum()
        real_count = (df['label'] == 0).sum()
        print(f"   {name}: Fake={fake_count} ({fake_count/len(df)*100:.1f}%), "
              f"Real={real_count} ({real_count/len(df)*100:.1f}%)")
    
    # 9. Check if it's a toy dataset
    print("\n9. TOY DATASET INDICATORS:")
    
    indicators = []
    
    if len(train_df) < 1000:
        indicators.append("✓ Very small dataset (<1000 samples)")
    
    if len(fake_only_words) > 20 or len(real_only_words) > 20:
        indicators.append("✓ Classes have completely distinct vocabularies")
    
    if train_test_overlap > 0:
        indicators.append("✓ Data leakage between train and test")
    
    avg_length = train_df['text'].str.len().mean()
    if avg_length < 100:
        indicators.append("✓ Very short texts")
    
    if len(vocab) < 1000:
        indicators.append(f"✓ Very small vocabulary ({len(vocab)} words)")
    
    if indicators:
        print("\n   ⚠️ WARNING: This appears to be a TOY/SYNTHETIC dataset!")
        for indicator in indicators:
            print(f"   {indicator}")
    else:
        print("   Dataset appears to be realistic")
    
    print("\n" + "="*60)
    print("INVESTIGATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    investigate_data()
