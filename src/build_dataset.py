"""CLI: raw BDG2 CSV -> cleaned, normalized, windowed train/val/test arrays.

Usage:
    python src/build_dataset.py --max-buildings 200
"""
import argparse
import json
import os

import numpy as np

from data_loader import load_electricity_wide, select_buildings, to_long
from preprocessing import (
    chronological_split,
    inject_synthetic_anomalies,
    interpolate_and_clean,
    make_windows,
    zscore_normalize,
)

OUT_DIR = "data/processed"


def main(max_buildings: int, max_missing_frac: float, window: int, anomaly_rate: float):
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading raw electricity data...")
    wide = load_electricity_wide()

    print(f"Selecting up to {max_buildings} buildings with <= {max_missing_frac:.0%} missing...")
    building_ids = select_buildings(wide, max_buildings=max_buildings, max_missing_frac=max_missing_frac)
    print(f"  -> {len(building_ids)} buildings selected")

    long_df = to_long(wide, building_ids)

    print("Interpolating short gaps / dropping buildings with long gaps...")
    long_df = interpolate_and_clean(long_df)
    kept = long_df["building_id"].nunique()
    print(f"  -> {kept} buildings kept after cleaning")

    print("Splitting chronologically (train/val/test)...")
    train_df, val_df, test_df = chronological_split(long_df)

    print("Z-score normalizing per building (fit on train only)...")
    train_norm, scalers = zscore_normalize(train_df)
    val_norm = _apply_scalers(val_df, scalers)
    test_norm = _apply_scalers(test_df, scalers)

    print(f"Injecting synthetic anomalies into test split (rate={anomaly_rate})...")
    test_norm["is_anomaly"] = False
    test_norm = inject_synthetic_anomalies(test_norm, anomaly_rate=anomaly_rate)

    print("Windowing...")
    train_windows = make_windows(train_norm, window=window, stride=1, has_labels=False)
    val_windows = make_windows(val_norm.assign(is_anomaly=False), window=window, stride=1, has_labels=False)
    test_windows = make_windows(test_norm, window=window, stride=window // 4, has_labels=True)

    print(f"  train: {train_windows['X'].shape}, val: {val_windows['X'].shape}, "
          f"test: {test_windows['X'].shape} (anomaly rate {test_windows['y'].mean():.3f})")

    np.savez_compressed(f"{OUT_DIR}/train.npz", X=train_windows["X"])
    np.savez_compressed(f"{OUT_DIR}/val.npz", X=val_windows["X"])
    np.savez_compressed(f"{OUT_DIR}/test.npz", X=test_windows["X"], y=test_windows["y"],
                         building_id=test_windows["building_id"])
    with open(f"{OUT_DIR}/scalers.json", "w") as f:
        json.dump(scalers, f)

    print("Saved processed arrays to", OUT_DIR)


def _apply_scalers(df, scalers):
    df = df.copy()
    mu = df["building_id"].map(lambda b: scalers[b]["mean"])
    sigma = df["building_id"].map(lambda b: scalers[b]["std"])
    df["value"] = (df["electricity"] - mu) / sigma
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-buildings", type=int, default=200)
    parser.add_argument("--max-missing-frac", type=float, default=0.05)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--anomaly-rate", type=float, default=0.03)
    args = parser.parse_args()
    main(args.max_buildings, args.max_missing_frac, args.window, args.anomaly_rate)
