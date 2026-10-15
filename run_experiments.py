"""One gated pass: feature bundle, C sweep, train-only OOF, optional tree.

Input: the three CSVs. Does not read a test label.
Output: outputs/metrics_experiments.json.
        outputs/submission_candidate.csv only if the winner beats the baseline
        by more than MIN_GAIN and the OOF check passes.
Depends on: config, data_io, features, modeling, threshold, submission.

Run run_baseline.py first. This script does not overwrite the baseline or LightGBM files.
A format-checked candidate file is also left in place.
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
    parser = argparse.ArgumentParser(description="Gated experiments against the locked baseline.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--hgb",
        action="store_true",
        help="Also try one HistGradientBoosting model. No blend.",
    )
    return parser.parse_args()


def _score_logit(x_train, y_train, x_val, y_val, x_test, bundle, C):
    model = modeling.make_logit(bundle, C)
    model.fit(x_train, y_train)
    chosen = threshold.select_threshold(y_val, modeling.positive_proba(model, x_val))
    return {
        "bundle": bundle,
        "C": C,
        "model": "logistic_regression",
        "test_proba": modeling.positive_proba(model, x_test),
        **chosen,
    }


def _oof_f1(train_df, y_train, bundle, C):
    """Preprocessing is refit inside each training fold. Val and test are not used."""
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
        model = modeling.make_logit(bundle, C)
        model.fit(
            features.build_features(fold_train, stats, bundle=bundle),
            y_train[train_idx],
        )
        oof[hold_idx] = modeling.positive_proba(
            model,
            features.build_features(fold_hold, stats, bundle=bundle),
        )
    chosen = threshold.select_threshold(y_train, oof)
    return chosen["f1"], chosen["threshold"]


def _prepare(train, val, test, bundle):
    stats = features.fit_module_stats(train)
    return (
        features.build_features(train, stats, bundle=bundle),
        features.build_features(val, stats, bundle=bundle),
        features.build_features(test, stats, bundle=bundle),
    )


def main(data_dir=None, run_hgb=False):
    existing = submission.checked_record(config.EXPERIMENT_METRICS, config.CANDIDATE_SUBMISSION)
    if existing:
        print("Keeping the checked candidate. outputs/submission_candidate.csv was not overwritten.")
        print(existing.get("format_report", ""))
        return existing

    train, val, test = data_io.load_splits(data_dir)
    y_train = train[config.LABEL].to_numpy()
    y_val = val[config.LABEL].to_numpy()
    test_ids = test[config.ID_COL].copy()

    base_train, base_val, base_test = _prepare(train, val, test, bundle=False)
    bundle_train, bundle_val, bundle_test = _prepare(train, val, test, bundle=True)
    baseline = _score_logit(
        base_train, y_train, base_val, y_val, base_test, False, config.BASE_C
    )
    bundled = _score_logit(
        bundle_train, y_train, bundle_val, y_val, bundle_test, True, config.BASE_C
    )
    use_bundle = bundled["f1"] - baseline["f1"] > config.MIN_GAIN
    print(
        f"A bundle val F1={bundled['f1']:.3f} baseline={baseline['f1']:.3f} "
        f"{'KEEP' if use_bundle else 'REVERT'}"
    )

    chosen_bundle = use_bundle
    if chosen_bundle:
        matrices = (bundle_train, bundle_val, bundle_test)
    else:
        matrices = (base_train, base_val, base_test)
    by_c = {
        C: _score_logit(matrices[0], y_train, matrices[1], y_val, matrices[2], chosen_bundle, C)
        for C in config.C_GRID
    }
    best_c = max(by_c, key=lambda value: by_c[value]["f1"])
    c_gain = by_c[best_c]["f1"] - by_c[config.BASE_C]["f1"]
    chosen_c = best_c if best_c != config.BASE_C and c_gain > config.MIN_GAIN else config.BASE_C
    print(
        f"B C={chosen_c} val F1={by_c[chosen_c]['f1']:.3f} "
        f"(best tried {best_c} gain={c_gain:.3f})"
    )

    oof_f1, oof_threshold = _oof_f1(train, y_train, chosen_bundle, chosen_c)
    val_f1 = by_c[chosen_c]["f1"]
    stable = abs(val_f1 - oof_f1) <= config.MIN_GAIN
    print(f"C OOF F1={oof_f1:.3f} val F1={val_f1:.3f} {'STABLE' if stable else 'REVERT'}")
    if stable:
        winner = by_c[chosen_c]
    else:
        chosen_bundle = False
        chosen_c = config.BASE_C
        winner = baseline

    if run_hgb:
        stats = features.fit_module_stats(train)
        x_train = features.build_features(train, stats, bundle=chosen_bundle)
        x_val = features.build_features(val, stats, bundle=chosen_bundle)
        x_test = features.build_features(test, stats, bundle=chosen_bundle)
        tree = modeling.make_hgb(chosen_bundle)
        tree.fit(x_train, y_train)
        tree_choice = threshold.select_threshold(y_val, modeling.positive_proba(tree, x_val))
        tree_gain = tree_choice["f1"] - winner["f1"]
        print(
            f"D hgb val F1={tree_choice['f1']:.3f} gain={tree_gain:.3f} "
            f"{'KEEP' if tree_gain > config.MIN_GAIN else 'REVERT'}"
        )
        if tree_gain > config.MIN_GAIN:
            winner = {
                "bundle": chosen_bundle,
                "C": None,
                "model": "hist_gradient_boosting",
                "test_proba": modeling.positive_proba(tree, x_test),
                **tree_choice,
            }

    record = {
        "baseline_f1": baseline["f1"],
        "bundle_f1": bundled["f1"],
        "used_bundle": bool(winner.get("bundle")),
        "chosen_C": winner.get("C"),
        "oof_f1": oof_f1,
        "oof_threshold": oof_threshold,
        "oof_stable": stable,
        "winner_model": winner["model"],
        "winner_f1": winner["f1"],
        "winner_threshold": winner["threshold"],
        "c_sweep": {str(key): value["f1"] for key, value in by_c.items()},
    }
    gain = winner["f1"] - baseline["f1"]
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if gain > config.MIN_GAIN:
        path = config.CANDIDATE_SUBMISSION
        submission.write_submission(test_ids, winner["test_proba"], winner["threshold"], path)
        ok, report = submission.check_submission(path, test_ids)
        record["submission"] = str(path)
        record["format_ok"] = ok
        record["format_report"] = report
        print(report)
        print(
            "Candidate does not replace the baseline or LightGBM files. "
            "Upload outputs/submission_candidate.csv only if the line above is TRANSMISSION VALID."
        )
        if not ok:
            config.EXPERIMENT_METRICS.write_text(json.dumps(record, indent=2))
            raise SystemExit(1)
    else:
        record["submission"] = None
        print(
            "No gated gain. Keep outputs/submission_baseline.csv. "
            "Candidate was not written."
        )

    config.EXPERIMENT_METRICS.write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, run_hgb=args.hgb)
