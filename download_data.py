#!/usr/bin/env python
"""
Script to download fake news datasets
Supports LIAR and other publicly available datasets
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import RAW_DATA_DIR
from src.utils import console

def download_file(url: str, dest_path: Path, chunk_size: int = 8192):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                pbar.update(len(chunk))

def download_liar():
    """Download LIAR dataset"""
    console.print("[cyan]Downloading LIAR dataset...[/cyan]")
    
    base_url = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/"
    files = [
        "train.tsv",
        "valid.tsv", 
        "test.tsv"
    ]
    
    output_dir = RAW_DATA_DIR / "liar"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        url = base_url + file
        dest = output_dir / file
        
        if dest.exists():
            console.print(f"[yellow]{file} already exists, skipping[/yellow]")
            continue
        
        console.print(f"Downloading {file}...")
        try:
            download_file(url, dest)
            console.print(f"[green]✓ Downloaded {file}[/green]")
        except Exception as e:
            console.print(f"[red]Error downloading {file}: {e}[/red]")
            console.print("[yellow]Trying alternative source...[/yellow]")
            
            # Alternative: Download from backup source
            alt_url = f"https://www.cs.ucsb.edu/~william/data/liar_dataset/{file}"
            try:
                download_file(alt_url, dest)
                console.print(f"[green]✓ Downloaded {file} from alternative source[/green]")
            except Exception as e2:
                console.print(f"[red]Failed to download {file}: {e2}[/red]")
    
    console.print("[green]✓ LIAR dataset download complete![/green]")

def download_isot():
    """Download ISOT Fake News dataset (alternative small dataset)"""
    console.print("[cyan]Downloading ISOT Fake News dataset...[/cyan]")
    
    # This is a simplified version - you'd need actual URLs
    console.print("[yellow]ISOT dataset requires manual download from:[/yellow]")
    console.print("https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php")
    console.print("\nAfter downloading:")
    console.print(f"1. Extract the files to: {RAW_DATA_DIR}")
    console.print("2. Run: python src/data_io.py --dataset custom")

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    console.print("[cyan]Creating sample dataset for testing...[/cyan]")
    
    # Sample data for testing the pipeline
    sample_data = {
        'text': [
            "Scientists discover new species of butterfly in Amazon rainforest after decade-long study.",
            "BREAKING: Aliens confirmed to be living among us, government finally admits!",
            "Local community raises funds for new library through successful crowdfunding campaign.",
            "You won't believe what this celebrity did! Shocking revelations inside!",
            "New study shows benefits of Mediterranean diet for heart health.",
            "URGENT: Share this before it's deleted! Government hiding the truth!",
            "City council approves budget for road improvements and infrastructure.",
            "Miracle cure discovered! Doctors hate this one simple trick!",
            "Research team publishes findings on climate change impact on coastal regions.",
            "Unbelievable discovery will change everything you know about history!",
        ] * 50,  # Repeat to create 500 samples
        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 50,
        'title': [
            "New Butterfly Species Found",
            "ALIENS CONFIRMED!!!",
            "Library Fundraiser Success", 
            "Celebrity SHOCKING News",
            "Mediterranean Diet Study",
            "URGENT TRUTH REVEALED",
            "Infrastructure Budget Approved",
            "MIRACLE CURE FOUND",
            "Climate Research Published",
            "HISTORY CHANGING DISCOVERY"
        ] * 50
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add some variation
    import random
    df['text'] = df['text'].apply(lambda x: x + f" Additional context {random.randint(1, 100)}.")
    
    output_path = RAW_DATA_DIR / "sample_data.csv"
    df.to_csv(output_path, index=False)
    
    console.print(f"[green]✓ Sample dataset created: {output_path}[/green]")
    console.print(f"  - {len(df)} samples")
    console.print(f"  - {(df['label'] == 0).sum()} real news")
    console.print(f"  - {(df['label'] == 1).sum()} fake news")

def main():
    parser = argparse.ArgumentParser(description="Download fake news datasets")
    parser.add_argument('--dataset', type=str, default='liar',
                       choices=['liar', 'isot', 'sample', 'all'],
                       help='Dataset to download')
    args = parser.parse_args()
    
    # Create raw data directory
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == 'liar' or args.dataset == 'all':
        download_liar()
    
    if args.dataset == 'isot' or args.dataset == 'all':
        download_isot()
    
    if args.dataset == 'sample' or args.dataset == 'all':
        create_sample_dataset()
    
    console.print("\n[bold green]✨ Data download complete![/bold green]")
    console.print("\nNext steps:")
    console.print("1. Run: python src/data_io.py --dataset [liar|custom]")
    console.print("2. Run: python src/preprocess.py")
    console.print("3. Explore data: jupyter notebook notebooks/01_eda.ipynb")

if __name__ == "__main__":
    main()
