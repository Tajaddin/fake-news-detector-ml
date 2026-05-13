# fake-news-detector-ml

End-to-end fake news classifier with classical baselines, a fine-tuned transformer, ensemble voting, SHAP and LIME explainability, MLflow tracking, and a Streamlit demo. Built across three milestones for a graduate ML course.

## The honest finding

The headline result of this project is a negative one. Trained on the standard ISOT-style fake-vs-real news corpus, logistic regression hits 99 to 100 percent test accuracy out of the box. Real-world performance does not match.

`diagnose_perfect_accuracy.py` runs a six-step data audit: dataset size check, train/test text overlap, label-leaking metadata (publisher names, datetime artifacts, URL fragments), random-label baseline, train vs test accuracy gap, and seed sensitivity. The audit confirms publisher and stylistic leakage drive most of the apparent accuracy. The README treats this as the headline finding, not a footnote.

| Check | Result |
|---|---|
| Dataset size | Single-digit thousands per class, low diversity |
| Random-label baseline | Above chance, signals leakage |
| Train vs test gap | Near zero, signals memorized dataset features |
| Publisher leak | Reuters and PolitiFact mark labels through dateline patterns |

The classification numbers in this repo are not generalizable claims about real-world fake-news detection. They are a teaching artifact about why you do not trust a 100 percent test score.

## What the pipeline does

| Stage | Module |
|---|---|
| Download | `download_data.py` pulls the ISOT-style corpus |
| Preprocess | `src/preprocess.py` cleans HTML, normalizes whitespace, strips bylines |
| Features | `src/features.py` builds TF-IDF and handcrafted stylistic features |
| Baselines | `src/models_baseline.py` trains Logistic Regression, SVM, Random Forest, XGBoost |
| Transformer | `milestone3_transformer_model.py` fine-tunes a HuggingFace classifier head |
| Ensemble | `milestone3_ensemble.py` runs soft-voting across baseline plus transformer |
| Explainability | `milestone3_explainability.py` writes SHAP and LIME per-instance attributions |
| Demo | `milestone3_main.py` (Streamlit) accepts a pasted article and returns prediction plus top SHAP tokens |

## Stack

Python, scikit-learn, XGBoost, HuggingFace Transformers, PyTorch, SHAP, LIME, MLflow, Streamlit, pandas, numpy.

## Run

Two-phase scripts ship with the repo:

```
bash setup.sh          # venv plus pip install
bash run_phase1.sh     # baseline pipeline (download, preprocess, train, eval)
bash run_phase2.sh     # transformer, ensemble, explainability
```

Or step by step:

```
python download_data.py
make train_baseline
python milestone3_transformer_model.py
python milestone3_ensemble.py
python milestone3_explainability.py
streamlit run milestone3_main.py
```

To reproduce the data-quality audit:

```
python diagnose_perfect_accuracy.py
```

## Repository layout

```
fake-news-detector-ml/
  src/
    config.py              Paths, seeds, hyperparameters
    data_io.py             Loader and split helper
    preprocess.py          Text cleaning pipeline
    features.py            TF-IDF plus stylistic features
    models_baseline.py     Four classical classifiers with grid search
    train.py               Baseline training entry point
    evaluate.py            Per-model metric reporter
    utils.py               MLflow logging helpers
  milestone3_main.py             Streamlit demo
  milestone3_transformer_model.py Fine-tunes a HF classifier head
  milestone3_ensemble.py         Soft-voting across baselines plus transformer
  milestone3_explainability.py   SHAP and LIME outputs
  diagnose_perfect_accuracy.py   Six-step data audit
  check_raw_data.py              Raw corpus integrity check
  investigate_data.py            Bias and overlap probes
  test_features.py               Feature pipeline tests
  notebooks/01_eda.ipynb         EDA notebook plus .py mirror
  reports/figs/                  Confusion matrix, ROC, PR, threshold, calibration plots
  run_phase1.sh, run_phase2.sh   Two-phase runners
  Makefile                       train_baseline, eval, clean targets
```

## Plots in `reports/figs/`

| File | What it shows |
|---|---|
| `confusion_matrix_logistic_regression_42.png` | Test confusion matrix at seed 42 |
| `roc_curve_logistic_regression_42.png` | ROC with AUC |
| `pr_curve_logistic_regression_42.png` | Precision-recall |
| `calibration_logistic_regression_42.png` | Reliability diagram |
| `threshold_analysis_logistic_regression_42.png` | Precision and recall vs decision threshold |
| `milestone3_explanation.png` | SHAP token attribution sample |

## License

MIT, see `LICENSE`.
