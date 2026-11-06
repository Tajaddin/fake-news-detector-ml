#!/usr/bin/env python
"""
Script to download fake news datasets
Supports LIAR and other publicly available datasets
"""

import os
import sys
import hashlib
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

LIAR_MIRROR = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/"
LIAR_ZIP = "https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip"
# SHA-256 of the files in the official LIAR release; the mirror serves the same bytes.
LIAR_SHA256 = {
    "train.tsv": "23ef8cb7ff57b25c941100d0744039cebc595d65a67ea190f72b330a658d8c36",
    "valid.tsv": "8dd3d7ed1a11e78ed7df090c9eddb0f91147f895850c78d49dd7b4e88fd09a16",
    "test.tsv": "ee41e2c741e0019216952625145af8c4aa8c8d3057ec68284b134267bb845e6f",
}
TIMEOUT = 60  # seconds to connect, and between bytes received

def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def download_file(url: str, dest_path: Path, chunk_size: int = 8192):
    """Download file with progress bar. HTTP errors raise instead of being saved."""
    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        try:
            with open(dest_path, 'wb') as file:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        file.write(chunk)
                        pbar.update(len(chunk))
        except BaseException:
            dest_path.unlink(missing_ok=True)
            raise

def _move_if_verified(part: Path, dest: Path, expected: str):
    actual = sha256_of(part)
    if actual != expected:
        part.unlink()
        raise ValueError(f"{dest.name}: SHA-256 {actual} does not match the LIAR release ({expected})")
    part.replace(dest)

def download_liar():
    """Download LIAR dataset, checking every file against the release checksums"""
    console.print("[cyan]Downloading LIAR dataset...[/cyan]")
    
    output_dir = RAW_DATA_DIR / "liar"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    missing = []
    for file, expected in LIAR_SHA256.items():
        dest = output_dir / file
        if dest.exists() and sha256_of(dest) == expected:
            console.print(f"[yellow]{file} already exists, skipping[/yellow]")
        else:
            missing.append(file)
    
    for file in list(missing):
        console.print(f"Downloading {file}...")
        part = output_dir / f"{file}.part"
        try:
            download_file(LIAR_MIRROR + file, part)
            _move_if_verified(part, output_dir / file, LIAR_SHA256[file])
        except (requests.RequestException, ValueError) as e:
            console.print(f"[red]Error downloading {file}: {e}[/red]")
            continue
        missing.remove(file)
        console.print(f"[green]✓ Downloaded {file}[/green]")
    
    if missing:
        console.print("[yellow]Trying the official LIAR archive...[/yellow]")
        archive_path = output_dir / "liar_dataset.zip.part"
        try:
            download_file(LIAR_ZIP, archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                for file in missing:
                    part = output_dir / f"{file}.part"
                    part.write_bytes(archive.read(file))
                    _move_if_verified(part, output_dir / file, LIAR_SHA256[file])
                    console.print(f"[green]✓ Extracted {file} from the official archive[/green]")
        finally:
            archive_path.unlink(missing_ok=True)
    
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
        try:
            download_liar()
        except (requests.RequestException, ValueError, KeyError, zipfile.BadZipFile) as e:
            console.print(f"[red]Failed to download LIAR: {e}[/red]")
            return 1
    
    if args.dataset == 'isot' or args.dataset == 'all':
        download_isot()
    
    if args.dataset == 'sample' or args.dataset == 'all':
        create_sample_dataset()
    
    console.print("\n[bold green]✨ Data download complete![/bold green]")
    console.print("\nNext steps:")
    console.print("1. Run: python src/data_io.py --dataset [liar|custom]")
    console.print("2. Run: python src/preprocess.py")
    console.print("3. Explore data: jupyter notebook notebooks/01_eda.ipynb")
    return 0

if __name__ == "__main__":
    # Windows consoles default to a legacy code page; the output uses symbols like ✓.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
