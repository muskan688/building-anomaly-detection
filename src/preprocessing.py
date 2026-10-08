"""Clean, normalize, split, and window BDG2 electricity data.

Training data stays unlabeled/"normal" (assumes real anomalies are rare in
raw BDG2 readings). Synthetic anomalies (spikes, drops, flatlines) are
injected only into the held-out test split so we have ground truth to
score recall / false-positive rate against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 12  # hours
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
RNG_SEED = 42


def interpolate_and_clean(long_df: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """Linearly interpolate gaps up to `max_gap` hours per building; drop buildings
    that still have missing values afterwards (gap too long to trust)."""
    out = []
    for bid, g in long_df.groupby("building_id", sort=False):
        g = g.set_index("timestamp").sort_index()
        g["electricity"] = g["electricity"].interpolate(limit=max_gap, limit_direction="both")
        if g["electricity"].isna().any():
            continue
        g["building_id"] = bid
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def zscore_normalize(long_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    scalers = {}
    parts = []
    for bid, g in long_df.groupby("building_id", sort=False):
        mu, sigma = g["electricity"].mean(), g["electricity"].std()
        sigma = sigma if sigma > 1e-6 else 1.0
        g = g.copy()
        g["value"] = (g["electricity"] - mu) / sigma
        scalers[bid] = {"mean": float(mu), "std": float(sigma)}
        parts.append(g)
    return pd.concat(parts, ignore_index=True), scalers


def chronological_split(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, val, test = [], [], []
    for _, g in long_df.groupby("building_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        n = len(g)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        train.append(g.iloc[:n_train])
        val.append(g.iloc[n_train:n_train + n_val])
        test.append(g.iloc[n_train + n_val:])
    return (pd.concat(train, ignore_index=True),
            pd.concat(val, ignore_index=True),
            pd.concat(test, ignore_index=True))


def inject_synthetic_anomalies(long_df: pd.DataFrame, anomaly_rate: float = 0.03,
                                seed: int = RNG_SEED) -> pd.DataFrame:
    """Injects spike / drop / flatline anomalies into `value`, adds `is_anomaly` flag."""
    rng = np.random.default_rng(seed)
    out = []
    for bid, g in long_df.groupby("building_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True).copy()
        g["is_anomaly"] = False
        n = len(g)
        n_events = max(1, int(n * anomaly_rate / 6))  # each event spans ~a few hours
        for _ in range(n_events):
            kind = rng.choice(["spike", "drop", "flatline"])
            start = rng.integers(0, max(1, n - 6))
            length = rng.integers(1, 6)
            end = min(n, start + length)
            if kind == "spike":
                g.loc[start:end - 1, "value"] += rng.uniform(3.0, 6.0)
            elif kind == "drop":
                g.loc[start:end - 1, "value"] -= rng.uniform(3.0, 6.0)
            else:  # flatline (stuck sensor)
                g.loc[start:end - 1, "value"] = g.loc[start, "value"]
            g.loc[start:end - 1, "is_anomaly"] = True
        g["building_id"] = bid
        out.append(g)
    return pd.concat(out, ignore_index=True)


def make_windows(long_df: pd.DataFrame, window: int = WINDOW, stride: int = 1,
                  has_labels: bool = False):
    """Slide a window per building. Returns X (N, window, 1), building_ids (N,),
    end_timestamps (N,), and optionally y (N,) = 1 if any point in the window is anomalous."""
    X, bids, ends, labels = [], [], [], []
    for bid, g in long_df.groupby("building_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        vals = g["value"].to_numpy()
        ts = g["timestamp"].to_numpy()
        anom = g["is_anomaly"].to_numpy() if has_labels else None
        for start in range(0, len(vals) - window + 1, stride):
            end = start + window
            X.append(vals[start:end])
            bids.append(bid)
            ends.append(ts[end - 1])
            if has_labels:
                labels.append(bool(anom[start:end].any()))
    X = np.asarray(X, dtype=np.float32)[..., None]
    result = {"X": X, "building_id": np.asarray(bids), "end_timestamp": np.asarray(ends)}
    if has_labels:
        result["y"] = np.asarray(labels, dtype=np.int8)
    return result
