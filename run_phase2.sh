#!/bin/bash

# Phase 2: Baseline Models Pipeline
echo "================================================"
echo "Running Phase 2: Baseline Models Pipeline"
echo "================================================"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate fake-news

# Step 1: Extract features
echo ""
echo "Step 1: Extracting features..."
echo "------------------------"
python src/features.py --save

# Step 2: Train baseline models
echo ""
echo "Step 2: Training baseline models..."
echo "------------------------"

# Train each model individually
echo "Training Logistic Regression..."
python src/train.py --model logistic_regression --experiment baseline_lr

echo ""
echo "Training SVM..."
python src/train.py --model svm --experiment baseline_svm

echo ""
echo "Training Naive Bayes..."
python src/train.py --model naive_bayes --experiment baseline_nb

echo ""
echo "Training Complement Naive Bayes..."
python src/train.py --model complement_nb --experiment baseline_cnb

# Step 3: Compare all models
echo ""
echo "Step 3: Comparing models..."
echo "------------------------"
python src/models_baseline.py --model all

# Step 4: Evaluate best model with plots
echo ""
echo "Step 4: Evaluating models..."
echo "------------------------"
python src/evaluate.py --all

# Generate detailed evaluation for best model
echo ""
echo "Generating detailed evaluation..."
python src/evaluate.py --model models/logistic_regression_42.pkl --plots --report

# Step 5: Run cross-validation for best model
echo ""
echo "Step 5: Running cross-validation..."
echo "------------------------"
python src/train.py --model logistic_regression --cv 3 --experiment cv_baseline

echo ""
echo "================================================"
echo "✨ Phase 2 Complete!"
echo "================================================"
echo ""
echo "Results:"
echo "  - Trained models: models/"
echo "  - Experiment logs: experiments/runs.csv"
echo "  - Evaluation plots: reports/figs/"
echo "  - Model comparison: experiments/model_comparison.csv"
echo ""
echo "Best performing model (based on F1):"
tail -1 experiments/model_comparison.csv
echo ""
echo "Next: Run Phase 3 (Transformers & Contributions)"
echo "  ./run_phase3.sh"
