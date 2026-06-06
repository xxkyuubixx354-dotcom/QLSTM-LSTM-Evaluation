# QLSTM vs LSTM — XAU/USD Gold Price Prediction

A comparative study of a **Quantum Long Short-Term Memory (QLSTM)** network against a classical **LSTM** baseline for forecasting XAU/USD (Gold) daily closing prices.

---

## Overview

This project investigates whether variational quantum circuits embedded inside LSTM gates can offer any advantage over a classical LSTM for financial time series forecasting. Both models predict the next-day log-return of gold prices, which is then used to reconstruct the USD price.

---

## Project Structure

```
QLSTM-LSTM-Evaluation/
├── QLSTM/
│   ├── qlstm.py                    # QLSTM model, training, and evaluation
│   ├── XAU_1d_data.csv             # Daily OHLC data (2004–2023)
│   ├── checkpoints/
│   │   └── best_qlstm_v2.pt        # Best QLSTM checkpoint
│   ├── training_metrics_v2.csv     # Per-epoch train/val loss
│   ├── qlstm eval.txt              # Final evaluation metrics
│   ├── qlstm_v2_results.png        # Train/val/test price predictions
│   ├── qlstm_residuals.png         # Residual analysis plots
│   ├── qlstm_regime_analysis.png   # MAE broken down by market regime
│   └── qlstm_rolling_performance.png
├── LSTM/
│   ├── lstm.py                     # Classical LSTM model, training, and evaluation
│   ├── XAU_1d_data.csv             # Same daily OHLC data
│   ├── xauusd_lstm_final.h5        # Saved Keras model
│   └── train_val_mae.csv           # Per-epoch train/val MAE
└── XAU_1d_data.csv                 # Root-level copy of the dataset
```

---

## Models

### QLSTM (PyTorch + PennyLane)

A custom LSTM cell where the four gates (input, forget, cell, output) are computed using **variational quantum circuits (VQCs)** instead of purely classical linear layers.

**Architecture:**
- 2 stacked QLSTM cells
- 8 qubits per VQC, 2 variational layers
- Data re-uploading: inputs are re-encoded at every quantum layer
- Two VQCs per cell — one for (input, forget) gates, one for (cell, output) gates
- Classical bypass connection to ensure stable gradient flow
- Skip connection from the last input timestep to the output
- Dropout (p=0.2) before the output layer

**Training:**
- Loss: Huber loss (δ=0.5)
- Optimizer: Adam (lr=0.001, weight decay=1e-4)
- Scheduler: Cosine Annealing with Warm Restarts (T₀=30)
- Gradient clipping: max norm 0.5
- Early stopping: patience of 30 epochs

### LSTM (TensorFlow / Keras)

A standard two-layer stacked LSTM used as the classical baseline.

**Architecture:**
- LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(32, ReLU) → Dense(1)

**Training:**
- Loss: MSE
- Optimizer: Adam (lr=0.001)
- Early stopping: patience of 10 epochs on validation loss

---

## Data & Features

**Dataset:** XAU/USD daily OHLC prices (2004–2023), sourced from `XAU_1d_data.csv`.

**Features engineered from OHLC:**

| Feature | Description |
|---|---|
| `LogReturn` | Daily log-return: log(Close / Close_prev) |
| `HL_range` | Daily volatility: log(High / Low) |
| `OC_ret` | Intraday direction: log(Close / Open) |
| `MA5_ret` | 5-day rolling mean of log-returns |
| `MA20_ret` | 20-day rolling mean of log-returns |

The LSTM additionally downloads live **macro features** via `yfinance`:
- `DXY_ret` — US Dollar Index log-return
- `TNX_change` — 10Y Treasury yield change
- `VIX_ret` — VIX fear index log-return

**Data split:** 80% train / 10% validation / 10% test (chronological, no shuffling)

**Sequence length:** 30 days (QLSTM), 60 days (LSTM)

---

## Results

### QLSTM — Test Set Metrics

| Metric | Train | Validation | Test |
|---|---|---|---|
| MSE (USD²) | 159.54 | 340.91 | 273.67 |
| RMSE (USD) | 12.63 | 18.46 | 16.54 |
| MAE (USD) | 8.58 | 12.92 | 12.15 |
| MAPE (%) | 0.79 | 0.75 | 0.67 |
| R² Score | 0.9987 | 0.9872 | 0.9547 |
| Directional Acc (%) | 47.30 | 46.91 | 47.66 |

### LSTM — Test Set Metrics

| Metric | Train | Validation | Test |
|---|---|---|---|
| MSE (USD²) | 178.27 | 261.01 | 1707.59 |
| RMSE (USD) | 13.35 | 16.16 | 41.32 |
| MAE (USD) | 9.05 | 12.07 | 23.73 |
| MAPE (%) | 0.78 | 0.65 | 0.81 |
| R² Score | 0.9988 | 0.9717 | 0.9960 |
| Directional Acc (%) | 46.95 | 49.26 | 44.75 |

### Head-to-Head Comparison (Test Set)

| Metric | QLSTM | LSTM | Winner |
|---|---|---|---|
| RMSE (USD) | 16.54 | 41.32 | QLSTM |
| MAE (USD) | 12.15 | 23.73 | QLSTM |
| MAPE (%) | 0.67 | 0.81 | QLSTM |
| R² Score | 0.9547 | 0.9960 | LSTM |
| Directional Acc (%) | 47.66 | 44.75 | QLSTM |

The QLSTM achieves significantly lower price errors on the test set (MAE ~12 vs ~24 USD), while the LSTM retains a higher R² score driven by its closer tracking of long-run price trends.

---

## Setup & Usage

### QLSTM

```bash
cd QLSTM
pip install torch pennylane scikit-learn pandas numpy matplotlib yfinance
python qlstm.py
```

### LSTM

```bash
cd LSTM
pip install tensorflow scikit-learn pandas numpy matplotlib yfinance
python lstm.py
```

Both scripts will train from scratch (or resume from a checkpoint in the case of QLSTM), save the best model, and generate evaluation plots.

---

## Requirements

| Package | Purpose |
|---|---|
| `torch` | QLSTM model and training |
| `pennylane` | Quantum circuit simulation |
| `tensorflow` / `keras` | Classical LSTM |
| `scikit-learn` | Scalers and metrics |
| `pandas`, `numpy` | Data processing |
| `matplotlib` | Plotting |
| `yfinance` | Macro feature download (LSTM) |
