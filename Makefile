# Makefile for Fake News Detection Project

.PHONY: help setup download data preprocess eda baseline transformer train evaluate demo clean

help:
	@echo "Fake News Detection Project - Available Commands:"
	@echo ""
	@echo "  make setup        Install dependencies and create environment"
	@echo "  make download     Download datasets"
	@echo "  make data         Process and split data"
	@echo "  make preprocess   Run text preprocessing"
	@echo "  make eda          Run exploratory data analysis"
	@echo "  make baseline     Train baseline models"
	@echo "  make transformer  Train transformer models"
	@echo "  make evaluate     Evaluate all models"
	@echo "  make demo         Run demo application"
	@echo "  make clean        Clean generated files"
	@echo ""
	@echo "  make all          Run complete pipeline"

setup:
	@echo "Setting up environment..."
	conda create -n fake-news python=3.10 -y
	conda activate fake-news && pip install -r requirements.txt
	@echo "✓ Setup complete"

download:
	@echo "Downloading datasets..."
	python download_data.py --dataset liar
	@echo "✓ Data downloaded"

data:
	@echo "Processing data..."
	python src/data_io.py --dataset liar --format csv
	@echo "✓ Data processed"

preprocess:
	@echo "Preprocessing text..."
	python src/preprocess.py --minimal
	python src/preprocess.py
	@echo "✓ Preprocessing complete"

eda:
	@echo "Running EDA..."
	jupyter nbconvert --to python notebooks/01_eda.ipynb --output-dir=temp/
	python temp/01_eda.py
	@echo "✓ EDA complete"

baseline:
	@echo "Training baseline models..."
	python src/train.py --model logistic_regression
	python src/train.py --model svm
	python src/train.py --model naive_bayes
	@echo "✓ Baseline training complete"

transformer:
	@echo "Training transformer model..."
	python src/train.py --model distilbert
	@echo "✓ Transformer training complete"

evaluate:
	@echo "Evaluating models..."
	python src/evaluate.py --all
	@echo "✓ Evaluation complete"

demo:
	@echo "Starting demo..."
	streamlit run app/demo_streamlit.py

clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .ipynb_checkpoints */.ipynb_checkpoints
	rm -rf temp/
	@echo "✓ Clean complete"

# Run complete pipeline
all: download data preprocess eda baseline transformer evaluate
	@echo "✨ Complete pipeline finished!"
