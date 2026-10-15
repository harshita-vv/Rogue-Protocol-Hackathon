"""Pick a hard 0/1 cutoff by validation F1.

Input: labels and positive probabilities.
Output: threshold, F1, precision, recall, predicted-positive count.
Depends on: config.py
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

import config


def threshold_grid():
    return np.linspace(
        config.THRESHOLD_MIN,
        config.THRESHOLD_MAX,
        config.THRESHOLD_STEPS,
    )


def select_threshold(y_true, proba):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    best = None
    for cutoff in threshold_grid():
        pred = (proba >= cutoff).astype(np.int8)
        score = f1_score(y_true, pred, zero_division=0)
        if best is None or score > best["f1"]:
            best = {
                "threshold": float(cutoff),
                "f1": float(score),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "predicted_positives": int(pred.sum()),
            }
    return best
