"""LightGBM challenger. Run only after the baseline CSV has been format-checked.

Input: train.csv, val.csv, test.csv, plus the checked baseline metrics.
Output: outputs/submission_lightgbm.csv and outputs/metrics_lightgbm.json.
Depends on: config, data_io, features, modeling, threshold, submission.

Uses the locked baseline columns, with module stats fit on train only.
Does not overwrite the baseline, the candidate, or a format-checked LightGBM file.
If LightGBM is not installed, this exits without writing. There is no blend.
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
    parser = argparse.ArgumentParser(description="Compare LightGBM with the checked logistic baseline.")
    parser.add_argument("--data-dir", default=None)
    return parser.parse_args()


def _best_logistic():
    baseline = submission.checked_record(config.BASELINE_METRICS, config.BASELINE_SUBMISSION)
    if baseline is None:
        raise SystemExit(
            "LightGBM was not started. Run run_baseline.py first and keep the checked CSV."
        )
    best_f1 = float(baseline["f1"])
    best_name = "logistic_baseline"
    best_file = "outputs/submission_baseline.csv"
    experiment = submission.checked_record(config.EXPERIMENT_METRICS, config.CANDIDATE_SUBMISSION)
    if experiment is not None and float(experiment.get("winner_f1", -1)) > best_f1:
        best_f1 = float(experiment["winner_f1"])
        best_name = str(experiment.get("winner_model", "experiment"))
        best_file = "outputs/submission_candidate.csv"
    return baseline, best_f1, best_name, best_file


def _score_line(name, record):
    print(
        f"{name:22} F1={record['f1']:.3f} P={record['precision']:.3f} "
        f"R={record['recall']:.3f} pred1={record['predicted_positives']} "
        f"threshold={record['threshold']:.2f}"
    )


def _oof_f1(train_df, y_train):
    """Preprocessing and LightGBM are refit inside each training fold."""
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
        model = modeling.fit_lightgbm(
            features.build_features(fold_train, stats, bundle=False),
            y_train[train_idx],
            bundle=False,
        )
        oof[hold_idx] = modeling.positive_proba(
            model,
            features.build_features(fold_hold, stats, bundle=False),
        )
    chosen = threshold.select_threshold(y_train, oof)
    return chosen["f1"], chosen["threshold"]


def main(data_dir=None):
    available, detail = modeling.lightgbm_available()
    if not available:
        print(f"LightGBM is not available ({detail}). Skipping it. The baseline file is unchanged.")
        return None

    existing = submission.checked_record(config.LGB_METRICS, config.LGB_SUBMISSION)
    if existing:
        print("Keeping the checked LightGBM file. outputs/submission_lightgbm.csv was not overwritten.")
        print(existing.get("format_report", ""))
        return existing

    baseline, best_f1, best_name, best_file = _best_logistic()
    print(f"LightGBM {detail}. Comparing against {best_name} val F1={best_f1:.3f}.")

    train, val, test = data_io.load_splits(data_dir)
    y_train = train[config.LABEL].to_numpy()
    y_val = val[config.LABEL].to_numpy()
    test_ids = test[config.ID_COL].copy()
    stats = features.fit_module_stats(train)
    x_train = features.build_features(train, stats, bundle=False)
    x_val = features.build_features(val, stats, bundle=False)
    x_test = features.build_features(test, stats, bundle=False)

    model = modeling.fit_lightgbm(x_train, y_train, bundle=False)
    chosen = threshold.select_threshold(y_val, modeling.positive_proba(model, x_val))
    oof_f1, oof_threshold = _oof_f1(train, y_train)
    gain = chosen["f1"] - best_f1
    stable = abs(chosen["f1"] - oof_f1) <= config.MIN_GAIN
    keep = gain > config.MIN_GAIN and stable

    print("Same baseline columns. Thresholds are chosen separately.")
    _score_line("logistic_baseline", baseline)
    _score_line("lightgbm", chosen)
    print(
        f"lightgbm OOF F1={oof_f1:.3f} threshold={oof_threshold:.2f} "
        f"gain_vs_{best_name}={gain:.3f} {'STABLE' if stable else 'UNSTABLE'} "
        f"{'KEEP' if keep else 'REVERT'}"
    )

    path = config.LGB_SUBMISSION
    submission.write_submission(
        test_ids,
        modeling.positive_proba(model, x_test),
        chosen["threshold"],
        path,
    )
    ok, report = submission.check_submission(path, test_ids)
    record = {
        "model": "lightgbm",
        "bundle": False,
        "lightgbm_version": detail,
        "reference_model": best_name,
        "reference_f1": best_f1,
        "gain": gain,
        "oof_f1": oof_f1,
        "oof_threshold": oof_threshold,
        "oof_stable": stable,
        "kept_as_winner": bool(keep and ok),
        **chosen,
        "submission": str(path),
        "format_ok": ok,
        "format_report": report,
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.LGB_METRICS.write_text(json.dumps(record, indent=2))
    print(report)
    if not ok:
        print("LightGBM file failed the format check. Upload is still the previous checked CSV.")
        raise SystemExit(1)
    if keep:
        print("Upload outputs/submission_lightgbm.csv. The baseline file was not changed.")
    else:
        print(f"LightGBM is not the upload. Keep {best_file}.")
    return record


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir)
