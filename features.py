"""Vectorized features. Module stats are fit on a train slice only.

Input: a raw log dataframe and stats from fit_module_stats.
Output: a column-selected model frame. bundle=False is the locked baseline.
Depends on: config.py
"""

import numpy as np
import pandas as pd

import config

_TONE_PREFIXES = (
    ("crew need not worry.", "reassure"),
    ("nothing to see.", "reassure"),
    ("all is well.", "reassure"),
    ("logging.", "neutral"),
    ("noted.", "neutral"),
)


def fit_module_stats(train_df):
    """Median and std of reading by module, using this train slice only."""
    stats = train_df.groupby("module", sort=False)["reading"].agg(
        median="median", std="std"
    )
    stats["std"] = stats["std"].fillna(1.0).replace(0, 1.0)
    return stats


def _parse_messages(message):
    parts = message.str.extract(
        r"^NINE:\s*(?P<body>.*?)\s*\[(?P<bracket>[^\]]+)\]\s*$",
        expand=True,
    )
    body = parts["body"].fillna("")
    bracket = parts["bracket"].fillna("")
    redacted = bracket.str.contains("REDACTED", case=False, na=False)
    parsed = bracket.str.extract(
        r"^(?P<br_mod>.*?)\s+(?P<br_val>-?\d+\.?\d*)\s+(?P<br_unit>\S+)$",
        expand=True,
    )
    body_l = body.str.lower().str.strip()
    tone = pd.Series("bare", index=message.index, dtype="object")
    event = body_l.copy()
    for prefix, name in _TONE_PREFIXES:
        hit = body_l.str.startswith(prefix, na=False)
        tone = tone.mask(hit, name)
        event = event.mask(hit, body_l.str.slice(len(prefix)).str.strip())
    return pd.DataFrame(
        {
            "br_mod": parsed["br_mod"].mask(redacted),
            "br_val": pd.to_numeric(parsed["br_val"], errors="coerce").mask(redacted),
            "tone": tone,
            "event": event,
        },
        index=message.index,
    )


def build_features(df, stats, bundle=False):
    """Build model columns. Does not read is_true_fault."""
    parsed = _parse_messages(df["message"])
    reading = df["reading"]
    br_val = parsed["br_val"]
    both_present = br_val.notna() & reading.notna()
    abs_diff = (br_val - reading).abs()
    med = df["module"].map(stats["median"])
    std = df["module"].map(stats["std"]).replace(0, np.nan).fillna(1.0)
    z = (reading - med) / std

    frame = pd.DataFrame(
        {
            "delay_ms": df["delay_ms"].to_numpy(),
            "retry_count": df["retry_count"].to_numpy(),
            "z": z.to_numpy(),
            "abs_z": np.abs(z.to_numpy()),
            "mod_mismatch": (
                parsed["br_mod"].notna() & (parsed["br_mod"] != df["module"])
            ).astype(np.int8).to_numpy(),
            "val_mismatch": (both_present & (abs_diff > config.VALUE_MISMATCH_EPS))
            .astype(np.int8)
            .to_numpy(),
            "abs_diff": abs_diff.to_numpy(),
            "reading_missing": reading.isna().astype(np.int8).to_numpy(),
            "module": df["module"].to_numpy(),
            "mode": df["mode"].to_numpy(),
            "status": df["status"].to_numpy(),
            "node": df["node"].to_numpy(),
            "tone": parsed["tone"].to_numpy(),
            "event": parsed["event"].to_numpy(),
        },
        index=df.index,
    )

    if bundle:
        severe = frame["event"].isin(config.SEVERE_EVENTS)
        benign = frame["event"].isin(config.BENIGN_EVENTS)
        severity = np.where(benign, "benign", np.where(severe, "severe", "boilerplate"))
        reassure = frame["tone"].eq("reassure").to_numpy()
        low, high = config.ABS_Z_EDGES
        abs_z = frame["abs_z"]
        bins = np.where(
            abs_z.isna(),
            "missing",
            np.where(abs_z < low, "low", np.where(abs_z < high, "mid", "high")),
        )
        frame["both_mismatch"] = (
            (frame["mod_mismatch"] == 1) & (frame["val_mismatch"] == 1)
        ).astype(np.int8)
        frame["reassure_x_severe"] = (reassure & (severity == "severe")).astype(np.int8)
        frame["reassure_x_boilerplate"] = (
            reassure & (severity == "boilerplate")
        ).astype(np.int8)
        frame["abs_z_bin"] = bins
        frame["event_severity"] = severity
        frame["module_x_status"] = (
            frame["module"].astype(str) + "|" + frame["status"].astype(str)
        )

    columns = config.numeric_columns(bundle) + config.categorical_columns(bundle)
    return frame.loc[:, columns]
