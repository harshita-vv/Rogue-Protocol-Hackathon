"""Decide what to do with a verdict. This file does not send, store, or book anything."""

import pandas as pd

FAULT_ACTIONS = ("notify_specialist", "remember", "schedule_repair", "follow_up")
FALSE_ALARM_ACTIONS = ("stand_down",)


def parse_verdict(value):
    """Return 1, 0, or None. None means the row cannot be acted on."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number not in (0.0, 1.0):
        return None
    return int(number)


def decide(verdict):
    """Map an identified verdict to action names. False alarms never schedule a repair."""
    if verdict == 1:
        return list(FAULT_ACTIONS)
    if verdict == 0:
        return list(FALSE_ALARM_ACTIONS)
    return []


def prediction_column(frame):
    for name in ("verdict", "prediction"):
        if name in frame.columns:
            return name
    if "is_true_fault" in frame.columns and "message" not in frame.columns:
        return "is_true_fault"
    return None


def attach_logs(verdicts, catalog):
    """Join Round 1 verdicts to ship logs. Catalog labels are ground truth, not predictions."""
    column = prediction_column(verdicts)
    if column is None or "log_id" not in verdicts.columns:
        raise ValueError("Verdict CSV needs log_id and verdict, prediction, or is_true_fault.")
    left = verdicts[["log_id", column]].copy()
    left["log_id"] = left["log_id"].map(lambda value: "" if pd.isna(value) else str(value).strip())
    left = left.rename(columns={column: "verdict"})
    right = catalog.copy()
    right["log_id"] = right["log_id"].map(lambda value: "" if pd.isna(value) else str(value).strip())
    if "is_true_fault" in right.columns:
        right = right.rename(columns={"is_true_fault": "ground_truth"})
    return left.merge(right, on="log_id", how="left", indicator=True)
