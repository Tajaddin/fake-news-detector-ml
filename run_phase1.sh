#!/bin/bash

# Phase 1: Data & EDA Pipeline
echo "================================================"
echo "Running Phase 1: Data & EDA Pipeline"
echo "================================================"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate fake-news

# Step 1: Download data
echo ""
echo "Step 1: Downloading data..."
echo "------------------------"
python download_data.py --dataset sample
# For real data, use: python download_data.py --dataset liar

# Step 2: Load and split data
echo ""
echo "Step 2: Loading and splitting data..."
echo "------------------------"
python src/data_io.py --dataset custom --format csv

# Step 3: Preprocess data
echo ""
echo "Step 3: Preprocessing data..."
echo "------------------------"
# Minimal preprocessing for transformers
python src/preprocess.py --minimal

# Full preprocessing for traditional ML
python src/preprocess.py

# Step 4: Run EDA notebook (optional - opens Jupyter)
echo ""
echo "Step 4: Exploratory Data Analysis"
echo "------------------------"
echo "To run EDA interactively:"
echo "  jupyter notebook notebooks/01_eda.ipynb"

# Or convert notebook to Python and run
jupyter nbconvert --to python notebooks/01_eda.ipynb
python notebooks/01_eda.py

echo ""
echo "================================================"
echo "✨ Phase 1 Complete!"
echo "================================================"
echo ""
echo "Results:"
echo "  - Raw data: data/raw/"
echo "  - Processed data: data/processed/"
echo "  - EDA figures: reports/figs/"
echo ""
echo "Next: Run Phase 2 (Baseline Models)"
echo "  ./run_phase2.sh"
