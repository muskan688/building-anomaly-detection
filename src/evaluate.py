"""Evaluate the trained LSTM autoencoder against a synthetic-anomaly test set,
and compare it to an Isolation Forest baseline on the same windows.

Usage:
    python src/evaluate.py --target-fpr 0.045
"""
import argparse

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from model import LSTMAutoencoder

DATA_DIR = "data/processed"
MODEL_PATH = "models/lstm_autoencoder.pt"


def load_split(name: str):
    d = np.load(f"{DATA_DIR}/{name}.npz")
    return d


def reconstruction_errors(model, X: torch.Tensor, device, batch_size: int = 512) -> np.ndarray:
    """Max per-timestep squared error within each window, so a brief anomaly
    isn't averaged away by the surrounding normal hours."""
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(device)
            recon = model(batch)
            se = ((recon - batch) ** 2).mean(dim=2)  # (batch, seq_len)
            errs.append(se.max(dim=1).values.cpu().numpy())
    return np.concatenate(errs)


def threshold_for_fpr(val_errors: np.ndarray, target_fpr: float) -> float:
    """val split is assumed anomaly-free, so target_fpr-th percentile of its
    reconstruction error gives a threshold that yields ~target_fpr false-positive rate."""
    return float(np.quantile(val_errors, 1 - target_fpr))


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    fpr = ((y_pred == 1) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())
    auc = roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else float("nan")
    print(f"[{name}] precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} "
          f"fpr={fpr:.3f} auc={auc:.3f}")


def main(target_fpr: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = LSTMAutoencoder(seq_len=ckpt["seq_len"], hidden_dim=ckpt["hidden_dim"],
                             latent_dim=ckpt["latent_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    val = load_split("val")
    test = load_split("test")
    X_val = torch.from_numpy(val["X"])
    X_test = torch.from_numpy(test["X"])
    y_test = test["y"]

    print("Scoring with LSTM autoencoder...")
    val_errors = reconstruction_errors(model, X_val, device)
    test_errors = reconstruction_errors(model, X_test, device)
    thresh = threshold_for_fpr(val_errors, target_fpr)
    lstm_pred = (test_errors > thresh).astype(int)
    report("LSTM-AE", y_test, lstm_pred, test_errors)

    print("Scoring with Isolation Forest baseline...")
    X_flat_val = X_val.numpy().reshape(len(X_val), -1)
    X_flat_test = X_test.numpy().reshape(len(X_test), -1)
    iso = IsolationForest(n_estimators=200, contamination=target_fpr, random_state=42)
    iso.fit(X_flat_val)
    iso_scores = -iso.score_samples(X_flat_test)  # higher = more anomalous
    iso_val_scores = -iso.score_samples(X_flat_val)
    iso_thresh = threshold_for_fpr(iso_val_scores, target_fpr)
    iso_pred = (iso_scores > iso_thresh).astype(int)
    report("IsolationForest", y_test, iso_pred, iso_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-fpr", type=float, default=0.045)
    args = parser.parse_args()
    main(args.target_fpr)
