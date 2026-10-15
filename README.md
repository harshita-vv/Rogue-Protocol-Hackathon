# Rogue Protocol files for Kaggle

Cursor does not train these models. Upload this folder into a Kaggle notebook and run it there, with `train.csv`, `val.csv`, and `test.csv` attached.

Libraries already on Kaggle: pandas, numpy, scikit-learn.

## Order

1. Upload a valid starter CSV and leave it in place.
2. `python run_baseline.py --data-dir /kaggle/input/<your-dataset>`
3. If the script prints `TRANSMISSION VALID`, you may upload `outputs/submission_baseline.csv`.
4. `python run_experiments.py --data-dir /kaggle/input/<your-dataset>`
5. Upload `outputs/submission_candidate.csv` only if the script writes it and the format check passes. Otherwise keep the baseline file.
6. Only with time left: `python run_experiments.py --data-dir /kaggle/input/<your-dataset> --hgb`

`run_experiments.py` does not overwrite `submission_baseline.csv`. There is no ensemble.

## Files

| File | Needs | Writes |
|---|---|---|
| `config.py` | nothing | nothing |
| `data_io.py` | the three CSVs | three dataframes in memory |
| `features.py` | a dataframe and train-only module stats | model columns |
| `modeling.py` | feature columns and train labels | probabilities |
| `threshold.py` | val labels and probabilities | a hard threshold |
| `submission.py` | test ids in file order and probabilities | `log_id,is_true_fault` |
| `run_baseline.py` | the three CSVs | `outputs/submission_baseline.csv` |
| `run_experiments.py` | the three CSVs | `outputs/metrics_experiments.json`, and a candidate CSV only if it earns the gain |

A change is kept only when validation F1 rises by more than 0.02 and train-only out-of-fold F1 is within 0.02 of that validation F1.
