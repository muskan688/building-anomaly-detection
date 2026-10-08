# Real-Time Anomaly Detection for Building Energy

LSTM-autoencoder anomaly detection on 2 years of hourly electricity meter data
from 1,600+ non-residential buildings (BDG2 — Building Data Genome Project 2).

## Status

- [x] Data pipeline (load, clean, normalize, chronological split, windowing)
- [x] LSTM autoencoder model + training loop
- [x] Evaluation vs. Isolation Forest baseline (synthetic-anomaly test set)
- [ ] Simulated streaming pipeline
- [ ] Drift detection + automated retraining (MLOps loop)

## Data

Source: [buds-lab/building-data-genome-project-2](https://github.com/buds-lab/building-data-genome-project-2)
(`data/meters/cleaned/electricity_cleaned.csv` + `data/metadata/metadata.csv`).
Not committed to this repo (see `.gitignore`) — re-download with:

```bash
mkdir -p data/raw
curl -L -o data/raw/electricity_cleaned.csv \
  "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/meters/cleaned/electricity_cleaned.csv"
curl -L -o data/raw/metadata.csv \
  "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/metadata/metadata.csv"
```

## Approach

Real building-energy datasets have no labeled anomalies, so the model is
trained unsupervised on raw (assumed-normal) readings. Synthetic anomalies
(spikes, drops, flatlined/stuck sensors) are injected only into the held-out
test split, giving ground truth to score recall and false-positive rate
against — a standard approach in the anomaly-detection literature.

1. **`src/build_dataset.py`** — loads the widest, most complete subset of
   buildings, interpolates short gaps, z-score normalizes per building
   (fit on train only), splits chronologically (70/15/15), and slides
   24-hour windows. Synthetic anomalies are injected into the test split.
2. **`src/model.py`** — `LSTMAutoencoder`: an LSTM encoder compresses each
   24h window to a latent vector; an LSTM decoder reconstructs it.
   Anomaly score = reconstruction MSE.
3. **`src/train.py`** — trains on train-split windows (all assumed normal),
   early-stopping on val-split reconstruction loss.
4. **`src/evaluate.py`** — picks a detection threshold from the val-split
   error distribution to hit a target false-positive rate, then reports
   precision/recall/F1/AUC on the test split, alongside an Isolation Forest
   baseline trained on the same windows.

## Usage

```bash
python -m venv venv
venv/Scripts/activate          # or `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

cd src
python build_dataset.py --max-buildings 200
python train.py --epochs 15
python evaluate.py --target-fpr 0.045
```

## Next steps

- **Streaming simulation**: replay the test split as a timestamp-ordered
  event stream (e.g. via a simple generator or a queue), scoring each new
  window as it arrives and measuring end-to-end detection latency.
- **MLOps loop**: monitor the reconstruction-error distribution over time
  for drift (e.g. KS-test / PSI against the training distribution),
  trigger retraining automatically when drift is detected, and track
  runs/metrics (e.g. with MLflow, matching the pattern in the sibling
  [de-power-forecast](../de-power-forecast) project).
