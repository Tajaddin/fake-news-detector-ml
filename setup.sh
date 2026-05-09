#!/bin/bash

# Fake News Detection Project Setup Script
# This script sets up the complete environment for the project

echo "🚀 Setting up Fake News Detection Project..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment..."
conda create -n fake-news python=3.10 -y

# Activate environment (note: this won't persist after script ends)
eval "$(conda shell.bash hook)"
conda activate fake-news

# Install PyTorch with CUDA support (modify based on your system)
echo "🔧 Installing PyTorch..."
# For CPU only:
pip install torch torchvision torchaudio

# For CUDA 11.8 (uncomment if you have GPU):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "📥 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Create necessary directories
echo "📁 Creating project structure..."
mkdir -p data/raw data/processed
mkdir -p models experiments/mlruns
mkdir -p reports/figs
mkdir -p notebooks

# Initialize experiment tracking CSV
echo "📊 Initializing experiment tracking..."
echo "experiment_id,model_name,features,seed,accuracy,precision,recall,f1,auc,timestamp" > experiments/runs.csv

# Create a simple test to verify installation
echo "🧪 Testing installation..."
python -c "
import torch
import transformers
import shap
import lime
print('✅ PyTorch version:', torch.__version__)
print('✅ Transformers version:', transformers.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ All dependencies installed successfully!')
"

echo ""
echo "✨ Setup complete! ✨"
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate fake-news"
echo ""
echo "Next steps:"
echo "1. Download your dataset to data/raw/"
echo "2. Run the data preprocessing pipeline"
echo "3. Start with the baseline models"
echo ""
echo "Happy coding! 🎉"
