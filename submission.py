"""Write and check the upload file.

Input: test log_id in original file order, probabilities, threshold.
Output: CSV with exactly log_id,is_true_fault and integer 0/1 verdicts.
Depends on: pandas, numpy.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def checked_record(metrics_path, submission_path):
    """Return a metrics dict only when that submission already passed the format check."""
    metrics_path = Path(metrics_path)
    submission_path = Path(submission_path)
    if not metrics_path.is_file() or not submission_path.is_file():
        return None
    try:
        record = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("format_ok") is True:
        return record
    return None


def write_submission(log_ids, proba, threshold, path):
    log_ids = pd.Series(np.asarray(log_ids), copy=False)
    verdict = (np.asarray(proba) >= threshold).astype(np.int64)
    if len(verdict) != len(log_ids):
        raise ValueError(
            f"probability rows ({len(verdict)}) != log_id rows ({len(log_ids)})"
        )
    frame = pd.DataFrame({"log_id": log_ids.to_numpy(), "is_true_fault": verdict})
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def check_submission(path, test_ids):
    """Same rejection rules as the official checker, plus original id order."""
    test_ids = pd.Series(np.asarray(test_ids), copy=False)
    try:
        submitted = pd.read_csv(path)
    except Exception as exc:
        return False, f"TRANSMISSION REJECTED: could not read csv ({exc.__class__.__name__})"
    problems = []
    if list(submitted.columns) != ["log_id", "is_true_fault"]:
        problems.append(
            f"columns must be exactly log_id,is_true_fault (got {list(submitted.columns)})"
        )
    else:
        if submitted["log_id"].duplicated().any():
            problems.append(f"{int(submitted['log_id'].duplicated().sum())} duplicate log_ids")
        got = set(submitted["log_id"])
        expected = set(test_ids)
        missing = expected - got
        extra = got - expected
        if missing:
            problems.append(f"{len(missing)} test log_ids missing")
        if extra:
            problems.append(f"{len(extra)} log_ids not in test.csv")
        if not submitted["log_id"].reset_index(drop=True).equals(test_ids.reset_index(drop=True)):
            problems.append("log_id order does not match test.csv")
        if not submitted["is_true_fault"].isin([0, 1]).all():
            problems.append("every verdict must be exactly 1 or 0")
    if problems:
        return False, "TRANSMISSION REJECTED: " + "; ".join(problems)
    positives = int(submitted["is_true_fault"].sum())
    return True, f"TRANSMISSION VALID — {len(submitted)} verdicts, {positives} flagged as true faults"
