"""
Check raw data and preprocessing pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def check_raw_data():
    """Examine raw data files"""
    
    print("="*60)
    print("RAW DATA EXAMINATION")
    print("="*60)
    
    # Check what raw data files exist
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    print("\n1. RAW DATA FILES:")
    if raw_dir.exists():
        raw_files = list(raw_dir.glob("*"))
        for file in raw_files:
            print(f"   - {file.name} ({file.stat().st_size / 1024:.1f} KB)")
            
            # Try to load and examine each file
            if file.suffix == '.csv':
                try:
                    df = pd.read_csv(file)
                    print(f"     Shape: {df.shape}")
                    print(f"     Columns: {list(df.columns)}")
                    
                    # Check first few rows
                    print("\n     First 2 rows:")
                    for idx, row in df.head(2).iterrows():
                        text_col = 'text' if 'text' in df.columns else df.columns[0]
                        label_col = 'label' if 'label' in df.columns else df.columns[-1]
                        
                        text_preview = str(row[text_col])[:150] + "..."
                        print(f"     Row {idx}: Label={row[label_col]}")
                        print(f"              Text: {text_preview}")
                        
                except Exception as e:
                    print(f"     Error reading: {e}")
                    
            elif file.suffix == '.json':
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        print(f"     JSON array with {len(data)} items")
                        if data:
                            print(f"     First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                    elif isinstance(data, dict):
                        print(f"     JSON object with keys: {list(data.keys())[:5]}")
                except Exception as e:
                    print(f"     Error reading: {e}")
    else:
        print("   No raw data directory found!")
    
    print("\n2. PROCESSED DATA FILES:")
    if processed_dir.exists():
        processed_files = list(processed_dir.glob("*.csv"))
        for file in processed_files:
            print(f"   - {file.name}")
            
            # Load and check
            df = pd.read_csv(file)
            print(f"     Shape: {df.shape}")
            print(f"     Columns: {list(df.columns)}")
            
            # Check if preprocessing changed anything
            if 'text' in df.columns and 'text_processed' in df.columns:
                # Compare original vs processed
                for idx in range(min(2, len(df))):
                    orig = df.iloc[idx]['text']
                    proc = df.iloc[idx]['text_processed']
                    
                    if orig != proc:
                        print(f"\n     Example preprocessing (row {idx}):")
                        print(f"     Original: {orig[:100]}...")
                        print(f"     Processed: {proc[:100]}...")
                    else:
                        print(f"     Row {idx}: No preprocessing applied (text unchanged)")
    
    print("\n3. CRITICAL CHECKS:")
    
    # Load train data to do specific checks
    train_path = processed_dir / "train_processed.csv"
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        
        # Check if there are any special tokens or patterns
        print("\n   Checking for artificial patterns...")
        
        # Check if fake/real appears in the text itself
        if 'text' in train_df.columns:
            fake_texts = train_df[train_df['label'] == 1]['text']
            real_texts = train_df[train_df['label'] == 0]['text']
            
            # Check for template-like patterns
            fake_starts = fake_texts.str[:50].value_counts().head()
            real_starts = real_texts.str[:50].value_counts().head()
            
            if len(fake_starts) > 0 and fake_starts.iloc[0] > 5:
                print(f"   ⚠️ WARNING: Fake news texts have repeated patterns!")
                print(f"      Most common start: '{fake_starts.index[0]}' (appears {fake_starts.iloc[0]} times)")
            
            if len(real_starts) > 0 and real_starts.iloc[0] > 5:
                print(f"   ⚠️ WARNING: Real news texts have repeated patterns!")
                print(f"      Most common start: '{real_starts.index[0]}' (appears {real_starts.iloc[0]} times)")
            
            # Check for artificially inserted keywords
            fake_keywords = ['fake', 'hoax', 'conspiracy', 'aliens', 'miracle']
            real_keywords = ['report', 'official', 'government', 'study', 'research']
            
            for keyword in fake_keywords:
                count = fake_texts.str.lower().str.contains(keyword).sum()
                if count > len(fake_texts) * 0.5:  # If >50% contain this word
                    print(f"   ⚠️ Suspicious: '{keyword}' appears in {count}/{len(fake_texts)} fake samples")
            
            for keyword in real_keywords:
                count = real_texts.str.lower().str.contains(keyword).sum()
                if count > len(real_texts) * 0.5:  # If >50% contain this word
                    print(f"   ⚠️ Suspicious: '{keyword}' appears in {count}/{len(real_texts)} real samples")
    
    print("\n" + "="*60)

def check_config():
    """Check configuration settings"""
    print("\nCONFIGURATION CHECK:")
    print("="*60)
    
    try:
        import sys
        sys.path.append('src')
        from config import BASELINE_CONFIG, DATA_CONFIG
        
        print("\n1. TF-IDF Configuration:")
        if 'tfidf' in BASELINE_CONFIG:
            for key, value in BASELINE_CONFIG['tfidf'].items():
                print(f"   {key}: {value}")
        
        print("\n2. Data Configuration:")
        if DATA_CONFIG:
            for key, value in DATA_CONFIG.items():
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"   Could not load config: {e}")
    
    print("="*60)

if __name__ == "__main__":
    check_raw_data()
    check_config()
