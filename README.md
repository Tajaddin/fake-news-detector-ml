# fake-news-detector-ml

Fake-news classification built across three milestones for a graduate ML course: classical baselines on TF-IDF features, a fine-tuned DistilBERT, an ensemble (weighted average, majority vote and stacking), and LIME explanations.

## Result

| Test set | Model | Accuracy | ROC-AUC | F1 | Evidence |
|---|---|---|---|---|---|
| LIAR held-out test, 1,267 statements | TF-IDF (1-2 grams) + logistic regression | 62.1% | 0.676 | 0.561 | `experiments/holdout_baseline.json`; CI reruns it and `tests/test_eval_holdout.py` asserts it |

LIAR's six labels are binarized: true, mostly-true and half-true count as real; false, barely-true and pants-fire as fake. The model reads the statement and its context field but not the speaker, so it cannot score a statement by its speaker's track record.

## Where the 100% came from

`experiments/runs.csv` also logs a perfect 1.0 for every model. Those rows come from the synthetic sample set, not from real news. `run_phase1.sh` builds it with `download_data.py --dataset sample`: ten sentences, each repeated 50 times with a random "Additional context N." suffix. `run_phase2.sh` then trains the experiments named in those rows (`baseline_lr`, `baseline_svm`, `baseline_nb`, `baseline_cnb`, `cv_baseline`). With 500 rows made from ten sentences, every test sentence also appears in training, so any model scores 100%. Rerunning those steps reproduces 1.0 for all four models with the same selected hyperparameters (C = 0.01, alpha = 0.01).

The later rows in `runs.csv` (81-84% accuracy) and the plots in `reports/figs/` come from a 2,185-row test split of a dataset that is not in the repo, so they cannot be reproduced from a clone.

`diagnose_perfect_accuracy.py` is the audit for a score like that. On the sample set it flags the tiny dataset, test rows that appear verbatim in training, every test row matching a training row once the number suffix is ignored, class-exclusive vocabularies and repeated templates. Its shuffled-label baseline answers a different question, whether the model beats what it scores with no signal at all; permuting the labels hides a leak as well, so the 100% clears it.

| Check | What it looks for |
|---|---|
| Dataset size | fewer than 1,000 rows |
| Train/test overlap | test texts that also appear in training, verbatim or once case, digits and punctuation are ignored |
| Class vocabularies | words that occur in only one class |
| Templates | the same opening repeated more than five times |
| Shuffled-label baseline | the same model refit on 50 permutations of the training labels and scored on the true test labels: what it reaches with no signal |
| Real labels | train and test accuracy with the true labels; NO SIGNAL if test accuracy does not clear the baseline's 95th percentile |
| Keywords | single words present in most texts of one class |

## What the pipeline does

| Stage | Module |
|---|---|
| Download | `download_data.py` fetches LIAR and checks each file against the SHA-256 of the official release; `--dataset sample` writes the synthetic set |
| Split | `src/data_io.py` keeps LIAR's own train/valid/test split |
| Preprocess | `src/preprocess.py` strips HTML, URLs, e-mail addresses and @-mentions, normalizes whitespace |
| Features | `src/features.py` builds TF-IDF plus stylistic counts (length, punctuation, capitals, URLs, mentions) |
| Baselines | `src/models_baseline.py`: logistic regression, linear SVM, multinomial and complement naive Bayes, random forest, gradient boosting, with grid search |
| Held-out evaluation | `src/eval_holdout.py` |
| Transformer | `milestone3_transformer_model.py` fine-tunes `distilbert-base-uncased` |
| Ensemble | `milestone3_ensemble.py`: weighted average, majority vote, and stacking with a logistic-regression meta-learner trained on out-of-fold scores |
| Explainability | `milestone3_explainability.py`: LIME word weights and DistilBERT attention |
| Milestone 3 run | `milestone3_main.py` trains the transformer and the ensemble, then explains one test example |

`milestone3_main.py` has not been rerun on LIAR since the stacking fix. It fine-tunes DistilBERT six times (once per fold for the out-of-fold stacking scores, then on the full training set) and wants a GPU; on CPU it uses only the first 500 training rows.

## Stack

Python, scikit-learn, PyTorch, Hugging Face Transformers, LIME, pandas, numpy.

## How to run

Prerequisites: Python 3.11+ (CPU is enough for the baseline; the transformer phase wants CUDA).

```bash
# baseline LR pipeline (sklearn stack only, pinned with hashes, no torch)
pip install -r requirements-baseline.txt
python download_data.py --dataset liar
python src/data_io.py --dataset liar --format csv
python src/eval_holdout.py        # reproduces the 62.1% acc / 0.676 AUC result on the LIAR held-out test
pytest -q                         # unit tests plus the held-out regression check
```

Transformer, ensemble and explanation (after the LIAR split above):

```
pip install -r requirements.txt
python milestone3_main.py
```

To reproduce the 100% result and the audit on the synthetic sample set:

```
python download_data.py --dataset sample
python src/data_io.py --dataset custom --format csv
python src/preprocess.py
python diagnose_perfect_accuracy.py
```

`setup.sh`, `run_phase1.sh` and `run_phase2.sh` are the original milestone scripts (conda). `run_phase1.sh` builds the synthetic sample set, so their scores are 100% by construction.

## Repository layout

```
fake-news-detector-ml/
  src/
    config.py              Paths, seeds, hyperparameters
    data_io.py             Loader and split helper
    preprocess.py          Text cleaning pipeline
    features.py            TF-IDF plus stylistic features
    models_baseline.py     Classical classifiers with grid search
    train.py               Baseline training entry point
    evaluate.py            Per-model metric reporter
    eval_holdout.py        Held-out LIAR evaluation
    utils.py               Seeding, CSV experiment log, console helpers
  milestone3_main.py             Milestone 3 batch run
  milestone3_transformer_model.py Fine-tunes DistilBERT
  milestone3_ensemble.py         Weighted average, majority vote and stacking
  milestone3_explainability.py   LIME and attention outputs
  diagnose_perfect_accuracy.py   Data audit
  check_raw_data.py              Raw corpus integrity check
  investigate_data.py            Bias and overlap probes
  tests/                         pytest suite (download, preprocess, features, data_io, eval_holdout, ensemble, transformer inference, audit)
  test_features.py               Manual inspection script for a trained model (not a pytest test)
  notebooks/01_eda.ipynb         EDA notebook plus .py mirror
  reports/figs/                  Confusion matrix, ROC, PR, threshold, calibration plots
  requirements-baseline.txt      Hash-pinned baseline dependencies (used by CI)
  requirements.txt               Full pipeline dependencies
  run_phase1.sh, run_phase2.sh   Original milestone scripts (synthetic sample set)
  Makefile                       download, data, preprocess, baseline, evaluate and clean targets
```

## Plots in `reports/figs/`

The five logistic-regression plots come from the earlier 2,185-row dataset that is not in the repo (see above), not from LIAR.

| File | What it shows |
|---|---|
| `confusion_matrix_logistic_regression_42.png` | Test confusion matrix at seed 42 |
| `roc_curve_logistic_regression_42.png` | ROC with AUC |
| `pr_curve_logistic_regression_42.png` | Precision-recall |
| `calibration_logistic_regression_42.png` | Reliability diagram |
| `threshold_analysis_logistic_regression_42.png` | Precision and recall vs decision threshold |
| `milestone3_explanation.png` | LIME word weights and attention for one example |

## License

MIT, see `LICENSE`.
