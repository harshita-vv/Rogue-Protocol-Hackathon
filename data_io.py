"""Load the three CSVs once and reject a bad schema.

Input: a directory with train.csv, val.csv, and test.csv.
Output: three dataframes. Test log_id order is the file order.
Depends on: config.py
"""

import os
from pathlib import Path

import pandas as pd

import config


def discover_data_dir(explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("ROGUE_DATA_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(Path.cwd())
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(path for path in kaggle_input.iterdir() if path.is_dir())
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if all((resolved / name).is_file() for name in ("train.csv", "val.csv", "test.csv")):
            return resolved
    raise FileNotFoundError(
        "Could not find train.csv, val.csv, and test.csv. "
        "Pass --data-dir or set ROGUE_DATA_DIR."
    )


def _check_frame(name, frame):
    expected = config.EXPECTED_ROWS[name]
    if len(frame) != expected:
        raise ValueError(f"{name}.csv has {len(frame)} rows, expected {expected}")
    missing = [col for col in config.RAW_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{name}.csv is missing columns: {missing}")
    if frame[config.ID_COL].duplicated().any():
        raise ValueError(f"{name}.csv has duplicate log_id values")
    has_label = config.LABEL in frame.columns
    if name == "test" and has_label:
        raise ValueError("test.csv must not contain is_true_fault")
    if name != "test" and not has_label:
        raise ValueError(f"{name}.csv is missing is_true_fault")


def load_splits(data_dir=None):
    """Read each CSV once. Do not sort. Do not use a test label."""
    directory = discover_data_dir(data_dir)
    train = pd.read_csv(directory / "train.csv")
    val = pd.read_csv(directory / "val.csv")
    test = pd.read_csv(directory / "test.csv")
    _check_frame("train", train)
    _check_frame("val", val)
    _check_frame("test", test)
    return train, val, test
