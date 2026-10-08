"""Train the LSTM autoencoder on normal (unlabeled) windows.

Usage:
    python src/train.py --epochs 15
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import LSTMAutoencoder

DATA_DIR = "data/processed"
MODEL_PATH = "models/lstm_autoencoder.pt"


def load_split(name: str) -> torch.Tensor:
    arr = np.load(f"{DATA_DIR}/{name}.npz")["X"]
    return torch.from_numpy(arr)


def main(epochs: int, batch_size: int, lr: float, hidden_dim: int, latent_dim: int, patience: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train = load_split("train")
    X_val = load_split("val")
    seq_len = X_train.shape[1]

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False)

    model = LSTMAutoencoder(seq_len=seq_len, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch,) in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(X_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                val_loss += criterion(recon, batch).item() * batch.size(0)
        val_loss /= len(X_val)

        print(f"Epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({"state_dict": model.state_dict(), "seq_len": seq_len,
                        "hidden_dim": hidden_dim, "latent_dim": latent_dim}, MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    print(f"Best val loss: {best_val:.5f}. Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.lr, args.hidden_dim, args.latent_dim, args.patience)
