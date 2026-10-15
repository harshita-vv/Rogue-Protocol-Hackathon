"""Constants for the Kaggle run. No data access."""

from pathlib import Path

LABEL = "is_true_fault"
ID_COL = "log_id"

RAW_COLUMNS = [
    "log_id",
    "timestamp",
    "module",
    "mode",
    "node",
    "status",
    "reading",
    "delay_ms",
    "retry_count",
    "message",
]
EXPECTED_ROWS = {"train": 6000, "val": 1000, "test": 3000}

# Node rates shifted between train and val. Leave off unless a Kaggle run shows a real gain.
INCLUDE_NODE = False

C_GRID = (0.3, 1.0, 3.0)
BASE_C = 1.0
THRESHOLD_MIN = 0.15
THRESHOLD_MAX = 0.60
THRESHOLD_STEPS = 19
MIN_GAIN = 0.02

N_SPLITS = 5
SEED = 0

VALUE_MISMATCH_EPS = 0.05
ABS_Z_EDGES = (0.8, 1.3)

BENIGN_EVENTS = frozenset(
    {
        "duplicate of earlier alert",
        "sensor warming up",
        "self-test pattern",
        "simulated drill event",
        "scheduled calibration",
        "recovered after retry",
    }
)
SEVERE_EVENTS = frozenset(
    {
        "bus arbitration deadlock",
        "scrubber saturation",
        "dish actuator stalled",
        "cascade brownout detected",
        "overvolt on rail b",
        "phase lock lost",
        "heartbeat missing from 3 nodes",
        "breaker tripped twice",
        "seal breach suspected",
        "queue overflow, packets dropped",
        "o2 partial pressure falling",
        "carrier drift beyond tolerance",
    }
)

OUTPUT_DIR = Path("outputs")
BASELINE_SUBMISSION = OUTPUT_DIR / "submission_baseline.csv"
BASELINE_METRICS = OUTPUT_DIR / "metrics_baseline.json"
CANDIDATE_SUBMISSION = OUTPUT_DIR / "submission_candidate.csv"
EXPERIMENT_METRICS = OUTPUT_DIR / "metrics_experiments.json"
LGB_SUBMISSION = OUTPUT_DIR / "submission_lightgbm.csv"
LGB_METRICS = OUTPUT_DIR / "metrics_lightgbm.json"

# One fixed challenger. No search and no blend with logistic regression.
LGB_PARAMS = {
    "objective": "binary",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}


def numeric_columns(bundle):
    cols = [
        "delay_ms",
        "retry_count",
        "z",
        "abs_z",
        "mod_mismatch",
        "val_mismatch",
        "abs_diff",
        "reading_missing",
    ]
    if bundle:
        cols.extend(
            ["both_mismatch", "reassure_x_severe", "reassure_x_boilerplate"]
        )
    return cols


def categorical_columns(bundle):
    cols = ["module", "mode", "status", "tone", "event"]
    if INCLUDE_NODE:
        cols.append("node")
    if bundle:
        cols.extend(["abs_z_bin", "event_severity", "module_x_status"])
    return cols
