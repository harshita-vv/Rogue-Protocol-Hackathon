"""Locked logistic baseline. Run this on Kaggle after the starter CSV is uploaded.

Input: train.csv, val.csv, test.csv.
Output: outputs/submission_baseline.csv and outputs/metrics_baseline.json.
Depends on: config, data_io, features, modeling, threshold, submission.

Does not write submission_candidate.csv. Upload this file only after TRANSMISSION VALID,
and only in place of the starter upload.
"""

import argparse
import json

import config
import data_io
import features
import modeling
import submission
import threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Fit the locked baseline and write a fallback CSV.")
    parser.add_argument("--data-dir", default=None)
    return parser.parse_args()


def main(data_dir=None):
    train, val, test = data_io.load_splits(data_dir)
    test_ids = test[config.ID_COL].copy()
    stats = features.fit_module_stats(train)
    x_train = features.build_features(train, stats, bundle=False)
    x_val = features.build_features(val, stats, bundle=False)
    x_test = features.build_features(test, stats, bundle=False)

    model = modeling.make_logit(bundle=False, C=config.BASE_C)
    model.fit(x_train, train[config.LABEL].to_numpy())
    chosen = threshold.select_threshold(
        val[config.LABEL].to_numpy(),
        modeling.positive_proba(model, x_val),
    )
    path = config.OUTPUT_DIR / "submission_baseline.csv"
    submission.write_submission(
        test_ids,
        modeling.positive_proba(model, x_test),
        chosen["threshold"],
        path,
    )
    ok, report = submission.check_submission(path, test_ids)
    record = {
        "model": "logistic_regression",
        "bundle": False,
        "C": config.BASE_C,
        **chosen,
        "submission": str(path),
        "format_ok": ok,
        "format_report": report,
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "metrics_baseline.json").write_text(json.dumps(record, indent=2))
    print(
        f"baseline val F1={chosen['f1']:.3f} threshold={chosen['threshold']:.2f} "
        f"P={chosen['precision']:.3f} R={chosen['recall']:.3f} "
        f"pred1={chosen['predicted_positives']}"
    )
    print(report)
    if not ok:
        raise SystemExit(1)
    print(
        "Keep the starter upload until you replace it with this checked file: "
        "outputs/submission_baseline.csv"
    )
    return record


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir)
