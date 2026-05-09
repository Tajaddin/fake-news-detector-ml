# Fake News Detector — ML

An end-to-end machine learning pipeline for fake news detection. Implements classical ML baselines alongside a fine-tuned transformer classifier with explainability (SHAP/LIME) and Streamlit demo.

## Features

- Text preprocessing and feature engineering pipeline
- Baseline classifiers (Logistic Regression, SVM, Random Forest, XGBoost)
- Fine-tuned transformer model (HuggingFace Transformers)
- Ensemble voting across models
- SHAP and LIME explainability
- MLflow experiment tracking
- Streamlit interactive demo

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Download dataset
python download_data.py

# Run baseline training pipeline
make train_baseline

# Run transformer training
python milestone3_transformer_model.py

# Run ensemble
python milestone3_ensemble.py

# Explainability analysis
python milestone3_explainability.py

# Streamlit demo
streamlit run milestone3_main.py
```

## Project Structure

```
fakenews-detector-ml/
├── src/               # Core modules (config, preprocess, features, models, train, eval, utils)
├── data/              # Dataset (downloaded via download_data.py)
├── models/            # Saved model checkpoints
├── notebooks/         # Exploratory notebooks
├── reports/           # Evaluation reports and figures
├── experiments/       # Experiment configs
├── Makefile           # Build automation
└── requirements.txt
```

## License

MIT — see [LICENSE](LICENSE)
