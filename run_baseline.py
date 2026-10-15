"""Logistic regression submission.

Input: train.csv, val.csv, test.csv.
With INCLUDE_NODE, this writes outputs/submission_node.csv and
outputs/metrics_node.json. It does not read or write the checked baseline files
except to compare F1. The baseline CSV stays the fallback.

Depends on: config, data_io, features, modeling, threshold, submission.
"""

import argparse
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold

import config
import data_io
import features
import modeling
import submission
import threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Fit logistic regression and write a submission CSV.")
    parser.add_argument("--data-dir", default=None)
    return parser.parse_args()


def _oof_f1(train_df, y_train):
    """Same logistic model. Module stats are refit inside each training fold."""
    oof = np.zeros(len(train_df), dtype=float)
    splitter = StratifiedKFold(
        n_splits=config.N_SPLITS,
        shuffle=True,
        random_state=config.SEED,
    )
    for train_idx, hold_idx in splitter.split(train_df, y_train):
        fold_train = train_df.iloc[train_idx]
        fold_hold = train_df.iloc[hold_idx]
        stats = features.fit_module_stats(fold_train)
        model = modeling.make_logit(bundle=False, C=config.BASE_C)
        model.fit(
            features.build_features(fold_train, stats, bundle=False),
            y_train[train_idx],
        )
        oof[hold_idx] = modeling.positive_proba(
            model,
            features.build_features(fold_hold, stats, bundle=False),
        )
    chosen = threshold.select_threshold(y_train, oof)
    return chosen["f1"], chosen["threshold"]


def main(data_dir=None):
    if config.INCLUDE_NODE:
        submission_path = config.NODE_SUBMISSION
        metrics_path = config.NODE_METRICS
        label = "node"
    else:
        submission_path = config.BASELINE_SUBMISSION
        metrics_path = config.BASELINE_METRICS
        label = "baseline"

    existing = submission.checked_record(metrics_path, submission_path)
    if existing:
        print(f"Keeping the checked {label} file. {submission_path} was not overwritten.")
        print(existing.get("format_report", ""))
        return existing

    fallback = submission.checked_record(config.BASELINE_METRICS, config.BASELINE_SUBMISSION)
    train, val, test = data_io.load_splits(data_dir)
    y_train = train[config.LABEL].to_numpy()
    test_ids = test[config.ID_COL].copy()
    stats = features.fit_module_stats(train)
    x_train = features.build_features(train, stats, bundle=False)
    x_val = features.build_features(val, stats, bundle=False)
    x_test = features.build_features(test, stats, bundle=False)

    model = modeling.make_logit(bundle=False, C=config.BASE_C)
    model.fit(x_train, y_train)
    chosen = threshold.select_threshold(
        val[config.LABEL].to_numpy(),
        modeling.positive_proba(model, x_val),
    )
    oof_f1, oof_threshold = _oof_f1(train, y_train)
    if submission_path.resolve() == config.BASELINE_SUBMISSION.resolve() and config.INCLUDE_NODE:
        raise RuntimeError("Node run must not write the baseline CSV.")
    submission.write_submission(
        test_ids,
        modeling.positive_proba(model, x_test),
        chosen["threshold"],
        submission_path,
    )
    ok, report = submission.check_submission(submission_path, test_ids)
    gain = None if fallback is None else chosen["f1"] - float(fallback["f1"])
    stable = abs(chosen["f1"] - oof_f1) <= config.MIN_GAIN
    recommend_node = bool(
        config.INCLUDE_NODE
        and ok
        and gain is not None
        and gain > config.MIN_GAIN
        and stable
    )
    record = {
        "model": "logistic_regression",
        "bundle": False,
        "include_node": config.INCLUDE_NODE,
        "C": config.BASE_C,
        **chosen,
        "oof_f1": oof_f1,
        "oof_threshold": oof_threshold,
        "oof_stable": stable,
        "baseline_f1": None if fallback is None else float(fallback["f1"]),
        "gain_vs_baseline": gain,
        "recommend_node": recommend_node,
        "submission": str(submission_path),
        "format_ok": ok,
        "format_report": report,
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(record, indent=2))
    print(
        f"{label} val F1={chosen['f1']:.3f} threshold={chosen['threshold']:.3f} "
        f"P={chosen['precision']:.3f} R={chosen['recall']:.3f} "
        f"pred1={chosen['predicted_positives']} "
        f"OOF F1={oof_f1:.3f} OOF threshold={oof_threshold:.3f}"
    )
    if fallback is not None:
        print(f"checked baseline val F1={float(fallback['f1']):.3f} gain={gain:.3f}")
    else:
        print("No checked baseline metrics found. outputs/submission_baseline.csv was not changed.")
    print(report)
    if not ok:
        raise SystemExit(1)
    if config.INCLUDE_NODE:
        if recommend_node:
            print("RECOMMEND outputs/submission_node.csv. The baseline file was not changed.")
        else:
            print("RETAIN outputs/submission_baseline.csv. Node file is saved but is not the upload.")
    return record


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir)
