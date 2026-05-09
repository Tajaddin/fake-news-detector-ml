"""
Data I/O module for Fake News Detection
Handles data loading, splitting, and saving
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import json
import csv
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.table import Table
import zipfile
import tarfile

from config import (
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    LABEL_MAP
)
from utils import set_seed, console

class DataLoader:
    """Unified data loader for multiple fake news datasets"""
    
    def __init__(self, dataset_name: str = "liar"):
        """
        Initialize data loader
        
        Args:
            dataset_name: Name of dataset ('liar', 'fakenewsnet', 'custom')
        """
        self.dataset_name = dataset_name.lower()
        self.console = Console()
        set_seed(RANDOM_SEED)
    
    def load_liar_dataset(self, data_dir: Path) -> pd.DataFrame:
        """
        Load LIAR dataset
        
        The LIAR dataset contains short statements with labels:
        - Columns: [id, label, statement, subjects, speaker, job, state, party, 
                   barely_true_cnt, false_cnt, half_true_cnt, mostly_true_cnt, 
                   pants_on_fire_cnt, context]
        
        Args:
            data_dir: Directory containing LIAR dataset files
        
        Returns:
            DataFrame with unified format
        """
        all_data = []
        
        # LIAR uses TSV files
        for split in ['train', 'valid', 'test']:
            file_path = data_dir / f"{split}.tsv"
            if file_path.exists():
                # LIAR dataset columns
                columns = ['id', 'label', 'statement', 'subjects', 'speaker', 
                          'speaker_job', 'state', 'party', 'barely_true_cnt',
                          'false_cnt', 'half_true_cnt', 'mostly_true_cnt',
                          'pants_on_fire_cnt', 'context']
                
                df = pd.read_csv(file_path, sep='\t', header=None, 
                               names=columns, on_bad_lines='skip')
                
                # Map labels to binary (simplified)
                # true, mostly-true, half-true -> 0 (real)
                # false, barely-true, pants-fire -> 1 (fake)
                label_mapping = {
                    'true': 0,
                    'mostly-true': 0,
                    'half-true': 0,
                    'false': 1,
                    'barely-true': 1,
                    'pants-fire': 1
                }
                
                df['binary_label'] = df['label'].map(label_mapping)
                
                # Create unified format
                unified_df = pd.DataFrame({
                    'id': df['id'],
                    'text': df['statement'],
                    'label': df['binary_label'],
                    'original_label': df['label'],
                    'speaker': df['speaker'],
                    'context': df['context'],
                    'split': split if split != 'valid' else 'val'
                })
                
                all_data.append(unified_df)
                self.console.print(f"✓ Loaded {split}: {len(unified_df)} samples")
        
        if not all_data:
            raise FileNotFoundError(f"No LIAR dataset files found in {data_dir}")
        
        return pd.concat(all_data, ignore_index=True)
    
    def load_fakenewsnet_dataset(self, data_dir: Path) -> pd.DataFrame:
        """
        Load FakeNewsNet dataset
        
        Args:
            data_dir: Directory containing FakeNewsNet files
        
        Returns:
            DataFrame with unified format
        """
        all_data = []
        
        # FakeNewsNet structure: politifact/gossipcop with real/fake subdirs
        for source in ['politifact', 'gossipcop']:
            source_dir = data_dir / source
            if not source_dir.exists():
                continue
            
            for label_name in ['real', 'fake']:
                label_dir = source_dir / label_name
                if not label_dir.exists():
                    continue
                
                label = 0 if label_name == 'real' else 1
                
                # Each article is in a separate directory
                for article_dir in label_dir.iterdir():
                    if article_dir.is_dir():
                        news_file = article_dir / 'news_content.json'
                        if news_file.exists():
                            try:
                                with open(news_file, 'r', encoding='utf-8') as f:
                                    article = json.load(f)
                                
                                all_data.append({
                                    'id': article_dir.name,
                                    'text': article.get('text', ''),
                                    'title': article.get('title', ''),
                                    'label': label,
                                    'source': source,
                                    'url': article.get('url', ''),
                                    'author': article.get('authors', ''),
                                    'publish_date': article.get('publish_date', '')
                                })
                            except Exception as e:
                                self.console.print(f"[yellow]Warning: Error loading {news_file}: {e}[/yellow]")
        
        if not all_data:
            self.console.print("[yellow]No FakeNewsNet data found. Using alternative format...[/yellow]")
            # Try CSV format (simplified version often available)
            for file in data_dir.glob("*.csv"):
                df = pd.read_csv(file)
                if 'label' in df.columns and ('text' in df.columns or 'content' in df.columns):
                    text_col = 'text' if 'text' in df.columns else 'content'
                    unified_df = pd.DataFrame({
                        'id': df.index,
                        'text': df[text_col],
                        'title': df.get('title', ''),
                        'label': df['label'].map({'real': 0, 'fake': 1}) if df['label'].dtype == 'object' else df['label']
                    })
                    all_data.append(unified_df)
                    self.console.print(f"✓ Loaded {file.name}: {len(unified_df)} samples")
        
        if not all_data:
            raise FileNotFoundError(f"No FakeNewsNet data found in {data_dir}")
        
        return pd.DataFrame(all_data) if isinstance(all_data[0], dict) else pd.concat(all_data)
    
    def load_custom_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Load custom CSV dataset
        
        Expected columns: text/content, label/target
        
        Args:
            file_path: Path to CSV file
        
        Returns:
            DataFrame with unified format
        """
        df = pd.read_csv(file_path)
        
        # Find text column
        text_columns = ['text', 'content', 'article', 'statement', 'news', 'body']
        text_col = None
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            raise ValueError(f"No text column found. Expected one of: {text_columns}")
        
        # Find label column
        label_columns = ['label', 'target', 'class', 'is_fake', 'fake']
        label_col = None
        for col in label_columns:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            raise ValueError(f"No label column found. Expected one of: {label_columns}")
        
        # Convert labels to binary if needed
        if df[label_col].dtype == 'object':
            # Try to map string labels
            unique_labels = df[label_col].unique()
            if len(unique_labels) == 2:
                # Binary classification
                label_map = {}
                for label in unique_labels:
                    if label.lower() in ['fake', 'false', 'unreliable', '1', 1]:
                        label_map[label] = 1
                    else:
                        label_map[label] = 0
                df['binary_label'] = df[label_col].map(label_map)
            else:
                raise ValueError(f"Found {len(unique_labels)} unique labels. Expected binary classification.")
        else:
            df['binary_label'] = df[label_col]
        
        # Create unified format
        unified_df = pd.DataFrame({
            'id': df.index,
            'text': df[text_col],
            'title': df.get('title', ''),
            'label': df['binary_label']
        })
        
        # Add any additional columns
        for col in df.columns:
            if col not in [text_col, label_col, 'binary_label'] and col not in unified_df.columns:
                unified_df[col] = df[col]
        
        return unified_df
    
    def load_data(self) -> pd.DataFrame:
        """
        Main method to load data based on dataset name
        
        Returns:
            DataFrame with unified format
        """
        self.console.print(f"\n[bold cyan]Loading {self.dataset_name} dataset...[/bold cyan]")
        
        if self.dataset_name == "liar":
            data_path = RAW_DATA_DIR / "liar"
            if not data_path.exists():
                # Try to find LIAR files directly in raw folder
                if (RAW_DATA_DIR / "train.tsv").exists():
                    data_path = RAW_DATA_DIR
                else:
                    raise FileNotFoundError(
                        f"LIAR dataset not found. Please download from:\n"
                        "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip\n"
                        f"Extract to: {RAW_DATA_DIR}/liar/"
                    )
            return self.load_liar_dataset(data_path)
        
        elif self.dataset_name == "fakenewsnet":
            data_path = RAW_DATA_DIR / "fakenewsnet"
            if not data_path.exists():
                raise FileNotFoundError(
                    f"FakeNewsNet dataset not found. Please download from:\n"
                    "https://github.com/KaiDMML/FakeNewsNet\n"
                    f"Extract to: {RAW_DATA_DIR}/fakenewsnet/"
                )
            return self.load_fakenewsnet_dataset(data_path)
        
        else:
            # Look for any CSV file in raw directory
            csv_files = list(RAW_DATA_DIR.glob("*.csv"))
            if csv_files:
                self.console.print(f"Found CSV file: {csv_files[0].name}")
                return self.load_custom_csv(csv_files[0])
            else:
                raise FileNotFoundError(f"No data files found in {RAW_DATA_DIR}")
    
    def create_splits(
        self,
        df: pd.DataFrame,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits
        
        Args:
            df: Input DataFrame
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # If data already has splits, use them
        if 'split' in df.columns:
            train_df = df[df['split'] == 'train'].copy()
            val_df = df[df['split'] == 'val'].copy()
            test_df = df[df['split'] == 'test'].copy()
            
            if len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0:
                return train_df, val_df, test_df
        
        # Otherwise create new splits
        # First split: train+val vs test
        X = df.drop(columns=['label'])
        y = df['label']
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_ratio,
            random_state=RANDOM_SEED,
            stratify=y
        )
        
        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=RANDOM_SEED,
            stratify=y_temp
        )
        
        # Reconstruct DataFrames
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        return train_df, val_df, test_df
    
    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        format: str = 'csv'
    ):
        """
        Save data splits to disk
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            format: Save format ('csv' or 'parquet')
        """
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            train_df.to_csv(PROCESSED_DATA_DIR / 'train.csv', index=False)
            val_df.to_csv(PROCESSED_DATA_DIR / 'val.csv', index=False)
            test_df.to_csv(PROCESSED_DATA_DIR / 'test.csv', index=False)
        elif format == 'parquet':
            train_df.to_parquet(PROCESSED_DATA_DIR / 'train.parquet', index=False)
            val_df.to_parquet(PROCESSED_DATA_DIR / 'val.parquet', index=False)
            test_df.to_parquet(PROCESSED_DATA_DIR / 'test.parquet', index=False)
        
        # Save statistics
        stats = {
            'dataset': self.dataset_name,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'train_fake_ratio': (train_df['label'] == 1).mean(),
            'val_fake_ratio': (val_df['label'] == 1).mean(),
            'test_fake_ratio': (test_df['label'] == 1).mean(),
            'features': list(train_df.columns)
        }
        
        with open(PROCESSED_DATA_DIR / 'data_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Display summary
        self.display_split_statistics(train_df, val_df, test_df)
        
        self.console.print(f"\n[green]✓ Data splits saved to {PROCESSED_DATA_DIR}[/green]")
    
    def display_split_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):
        """Display statistics about data splits"""
        
        table = Table(title="Data Split Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Split", style="cyan", no_wrap=True)
        table.add_column("Total", style="green")
        table.add_column("Fake", style="yellow")
        table.add_column("Real", style="yellow")
        table.add_column("Fake %", style="red")
        
        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            total = len(df)
            fake = (df['label'] == 1).sum()
            real = (df['label'] == 0).sum()
            fake_pct = fake / total * 100
            
            table.add_row(
                name,
                str(total),
                str(fake),
                str(real),
                f"{fake_pct:.1f}%"
            )
        
        self.console.print(table)
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load previously processed and split data
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Try CSV first, then parquet
        for ext in ['csv', 'parquet']:
            train_path = PROCESSED_DATA_DIR / f'train.{ext}'
            val_path = PROCESSED_DATA_DIR / f'val.{ext}'
            test_path = PROCESSED_DATA_DIR / f'test.{ext}'
            
            if train_path.exists() and val_path.exists() and test_path.exists():
                if ext == 'csv':
                    train_df = pd.read_csv(train_path)
                    val_df = pd.read_csv(val_path)
                    test_df = pd.read_csv(test_path)
                else:
                    train_df = pd.read_parquet(train_path)
                    val_df = pd.read_parquet(val_path)
                    test_df = pd.read_parquet(test_path)
                
                self.console.print(f"[green]✓ Loaded processed data from {PROCESSED_DATA_DIR}[/green]")
                self.display_split_statistics(train_df, val_df, test_df)
                
                return train_df, val_df, test_df
        
        raise FileNotFoundError(f"No processed data found in {PROCESSED_DATA_DIR}")


def main():
    """Main function to run data loading pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and split fake news dataset")
    parser.add_argument('--dataset', type=str, default='liar',
                       choices=['liar', 'fakenewsnet', 'custom'],
                       help='Dataset to load')
    parser.add_argument('--format', type=str, default='csv',
                       choices=['csv', 'parquet'],
                       help='Output format')
    args = parser.parse_args()
    
    # Initialize loader
    loader = DataLoader(dataset_name=args.dataset)
    
    # Load data
    df = loader.load_data()
    console.print(f"\n[green]✓ Loaded {len(df)} total samples[/green]")
    
    # Create splits
    train_df, val_df, test_df = loader.create_splits(df)
    
    # Save splits
    loader.save_splits(train_df, val_df, test_df, format=args.format)
    
    console.print("\n[bold green]✨ Data loading complete![/bold green]")


if __name__ == "__main__":
    main()
