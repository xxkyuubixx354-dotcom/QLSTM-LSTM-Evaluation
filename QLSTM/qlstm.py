import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import yfinance as yf



#  1. Reproducibility + Checkpoints
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_qlstm_v2.pt")
RESUME_PATH     = CHECKPOINT_PATH

SEQ_LENGTH  = 30         
BATCH_SIZE  = 64
EPOCHS      = 100
HIDDEN_SIZE = 16
N_QUBITS    = 8
N_LAYERS    = 2


#  2. Load data and compute log-returns
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(SCRIPT_DIR, "XAU_1d_data.csv"), sep=";")
df = df[['Date', 'Open', 'High', 'Low', 'Close']]
df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
df.index = df.pop('Date')
df.index = df.index.normalize()   

# Same date range as LSTM baseline for a fairer comparison
df = df.loc['2004-09-15':'2023-03-22']

# Feature engineering: stationary, bounded signals
df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))   # daily return
df['HL_range']  = np.log(df['High'] / df['Low'])               # daily volatility
df['OC_ret']    = np.log(df['Close'] / df['Open'])             # intraday direction
df['MA5_ret']   = df['LogReturn'].rolling(5).mean()            # short-term momentum
df['MA20_ret']  = df['LogReturn'].rolling(20).mean()           # medium-term momentum

# Drop NaNs from self-computed features
df.dropna(inplace=True)

FEATURES = ['LogReturn', 'HL_range', 'OC_ret', 'MA5_ret', 'MA20_ret']

N_FEATURES = len(FEATURES)
print(f"Data loaded: {len(df)} rows, {N_FEATURES} features: {FEATURES}")


#  3. Scale features with StandardScaler

features_raw = df[FEATURES].values                 
prices_raw   = df['Close'].values                  

train_end = int(len(features_raw) * 0.8)

scaler = StandardScaler()
scaler.fit(features_raw[:train_end])
features_scaled = scaler.transform(features_raw)

# Target is the scaled LogReturn column (index 0)
returns_scaled = features_scaled[:, 0:1]

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i, 0])   # predict LogReturn only
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = create_sequences(features_scaled, seq_length=SEQ_LENGTH)

# 80/10/10 split so matches LSTM baseline 
q_80 = int(len(X) * 0.80)
q_90 = int(len(X) * 0.90)

X_train = torch.tensor(X[:q_80])
y_train = torch.tensor(y[:q_80])
X_val   = torch.tensor(X[q_80:q_90])
y_val   = torch.tensor(y[q_80:q_90])
X_test  = torch.tensor(X[q_90:])
y_test  = torch.tensor(y[q_90:])

# Stores the real prices aligned with each split

offset = SEQ_LENGTH 
prices_all = prices_raw[offset:]  
prices_train = prices_all[:q_80]
prices_val   = prices_all[q_80:q_90]
prices_test  = prices_all[q_90:]

# The price before each target (needed to reconstruct from return)
prev_prices_train = prices_raw[offset - 1 : offset - 1 + q_80]
prev_prices_val   = prices_raw[offset - 1 + q_80 : offset - 1 + q_90]
prev_prices_test  = prices_raw[offset - 1 + q_90 : offset - 1 + len(X)]

print(f"Train/Val/Test split: 80/10/10 — aligned with LSTM baseline")
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")


#  4. Quantum circuit 

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    # Data re-uploading: re-encode inputs at every variational layer
    for layer_w in weights:
        qml.AngleEmbedding(inputs * (np.pi / 2), wires=range(N_QUBITS), rotation='Y')
        for i in range(N_QUBITS):
            qml.RY(layer_w[i, 0], wires=i)
            qml.RZ(layer_w[i, 1], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


#  5. Quantum gate module
class QuantumGate(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super().__init__()
        self.weights = nn.Parameter(
            torch.rand(n_layers, n_qubits, 2) * 2 * np.pi
        )

    def forward(self, x):
        results = quantum_circuit(x, self.weights)
        return torch.stack(results, dim=1).float()


#  6. QLSTM cell (projects quantum output → hidden_size)
class QLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits=N_QUBITS):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits

        # Two VQCs: one for (input, forget) gates, one for (cell, output) gates
        self.vqc_if = QuantumGate(n_qubits)
        self.vqc_go = QuantumGate(n_qubits)
        self.fc_if  = nn.Linear(input_size + hidden_size, n_qubits)
        self.fc_go  = nn.Linear(input_size + hidden_size, n_qubits)
        self.proj_if = nn.Linear(n_qubits, hidden_size * 2)
        self.proj_go = nn.Linear(n_qubits, hidden_size * 2)

        # Classical bypass to ensure gradients flow even when VQC is stuck
        self.classical_gates = nn.Linear(input_size + hidden_size, hidden_size * 4)

    def forward(self, x, hidden):
        h, c = hidden
        combined = torch.cat([x, h], dim=1)

        if_out = self.proj_if(self.vqc_if(torch.tanh(self.fc_if(combined))))  # (batch, hidden*2)
        go_out = self.proj_go(self.vqc_go(torch.tanh(self.fc_go(combined))))  # (batch, hidden*2)
        gates  = torch.cat([if_out, go_out], dim=1) + self.classical_gates(combined)
        i_raw, f_raw, g_raw, o_raw = gates.chunk(4, dim=1)

        i = torch.sigmoid(i_raw)
        f = torch.sigmoid(f_raw)
        g = torch.tanh(g_raw)
        o = torch.sigmoid(o_raw)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

#  7. QLSTM model
class QLSTMModel(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden_size=HIDDEN_SIZE, n_qubits=N_QUBITS):
        super().__init__()
        self.hidden_size = hidden_size
        self.qlstm1  = QLSTMCell(input_size,   hidden_size, n_qubits)  # layer 1
        self.qlstm2  = QLSTMCell(hidden_size,  hidden_size, n_qubits)  # layer 2
        self.dropout = nn.Dropout(p=0.2)
        self.fc_out  = nn.Linear(hidden_size, 1)
        self.fc_skip = nn.Linear(input_size, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        h1 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c1 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        h2 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c2 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(x.shape[1]):
            h1, c1 = self.qlstm1(x[:, t, :], (h1, c1))
            h2, c2 = self.qlstm2(h1,          (h2, c2))
        return self.fc_out(self.dropout(h2)) + self.fc_skip(x[:, -1, :])


#  8. Training
model     = QLSTMModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-5)
loss_fn   = nn.HuberLoss(delta=0.5)  # smooth near zero (avoids L1 zero-gradient stall), better for large errors

train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training QLSTM v2 | seq={SEQ_LENGTH}, hidden={HIDDEN_SIZE},"
      f"qubits={N_QUBITS}, layers={N_LAYERS}")
print(f"Batches per epoch: {len(train_dl)}")

best_val_loss    = float('inf')
patience         = 30   
patience_counter = 0
start_epoch      = 0


if RESUME_PATH and os.path.exists(RESUME_PATH):
    checkpoint = torch.load(RESUME_PATH, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch   = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resumed from epoch {start_epoch} | Best val loss: {best_val_loss:.6f}")
else:
    print("Starting fresh training run")

METRICS_PATH = "training_metrics_v2.csv"
metrics_log  = []

for epoch in range(start_epoch, EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_idx, (xb, yb) in enumerate(train_dl):
        pred = model(xb)
        p, t = pred.squeeze(), yb.squeeze()
        loss = loss_fn(p, t)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # ← tighter clip
        optimizer.step()
        epoch_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"   Epoch {epoch+1} | Batch {batch_idx}/{len(train_dl)} "
                  f"| Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = loss_fn(val_pred.squeeze(), y_val.squeeze()).item()

    train_mse = epoch_loss / len(train_dl)
    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS}"
          f"Train MSE: {train_mse:.6f}  Val MSE: {val_loss:.6f}")

    metrics_log.append({
        'epoch':         epoch + 1,
        'train_mse':     train_mse,
        'val_mse':       val_loss,
        'best_val_loss': best_val_loss,
        'lr':            optimizer.param_groups[0]['lr'],
    })

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save({
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_loss':   best_val_loss,
        }, CHECKPOINT_PATH)
        print(f"💾 Checkpoint saved (val MSE: {best_val_loss:.6f})")
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

pd.DataFrame(metrics_log).to_csv(METRICS_PATH, index=False)
print(f"Metrics saved to {METRICS_PATH}")

checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
model.load_state_dict(checkpoint['model_state'])
print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

# Diagnostic: confirm model is predicting non-zero returns
with torch.no_grad():
    _diag = model(X_val).numpy().flatten()
print(f"Predicted return std (val): {_diag.std():.4f}  mean: {_diag.mean():.4f}")


#  9. Predict log-returns → reconstruct prices
model.eval()
with torch.no_grad():
    pred_train_scaled = model(X_train).numpy()
    pred_val_scaled   = model(X_val).numpy()
    pred_test_scaled  = model(X_test).numpy()

# Inverse-transform scaled returns back to real log-returns
# Inverse-transform only the LogReturn column (index 0) as scaler covers all 5 features
def inv_transform_return(scaled_pred):
    dummy = np.zeros((len(scaled_pred), N_FEATURES), dtype=np.float32)
    dummy[:, 0] = scaled_pred.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

pred_train_lr = inv_transform_return(pred_train_scaled)
pred_val_lr   = inv_transform_return(pred_val_scaled)
pred_test_lr  = inv_transform_return(pred_test_scaled)

# Reconstruct prices
pred_train_prices = prev_prices_train * np.exp(pred_train_lr)
pred_val_prices   = prev_prices_val   * np.exp(pred_val_lr)
pred_test_prices  = prev_prices_test  * np.exp(pred_test_lr)


#  Evaluation metrics (matches LSTM baseline for direct comparison)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics(y_actual, y_predicted, set_name=''):
    y_act  = y_actual.flatten()
    y_pred = y_predicted.flatten()

    mse  = mean_squared_error(y_act, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_act, y_pred)
    mape = np.mean(np.abs((y_act - y_pred) / y_act)) * 100
    r2   = r2_score(y_act, y_pred)

    actual_dir = np.diff(y_act)
    pred_dir   = np.diff(y_pred)
    dir_acc    = np.mean(np.sign(actual_dir) == np.sign(pred_dir)) * 100

    return {
        'Set':                 set_name,
        'MSE (USD²)':          f'{mse:.4f}',
        'RMSE (USD)':          f'{rmse:.4f}',
        'MAE (USD)':           f'{mae:.4f}',
        'MAPE (%)':            f'{mape:.2f}',
        'R² Score':            f'{r2:.4f}',
        'Directional Acc (%)': f'{dir_acc:.2f}',
    }

train_metrics = compute_metrics(prices_train,     pred_train_prices, 'Train')
val_metrics   = compute_metrics(prices_val,       pred_val_prices,   'Validation')
test_metrics  = compute_metrics(prices_test,      pred_test_prices,  'Test')

metrics_df = pd.DataFrame([train_metrics, val_metrics, test_metrics]).set_index('Set')
print('\n' + '='*70)
print('              QLSTM MODEL EVALUATION METRICS')
print('='*70)
print(metrics_df.to_string())
print('='*70)


#  10. Plot results
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, preds, actual, title in zip(
    axes,
    [pred_train_prices, pred_val_prices, pred_test_prices],
    [prices_train,      prices_val,      prices_test],
    ['Train',           'Validation',    'Test'],
):
    ax.plot(actual, label='Actual',  linewidth=1.5)
    ax.plot(preds,  label='QLSTM',   linewidth=1.5, alpha=0.8)
    ax.set_title(f'{title} Set')
    ax.set_ylabel('Gold Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('QLSTM v2 — XAU/USD Gold Price (Log-Return Approach)', fontsize=14)
plt.tight_layout()
plt.savefig("qlstm_v2_results.png", dpi=150)
plt.show()


#  11. Residual Analysis
test_residuals = prices_test.flatten() - pred_test_prices.flatten()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Residuals over time
axes[0].plot(test_residuals, color='coral', linewidth=1)
axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[0].fill_between(range(len(test_residuals)), test_residuals, 0,
                     where=(test_residuals > 0), alpha=0.3, color='green', label='Over-predicted')
axes[0].fill_between(range(len(test_residuals)), test_residuals, 0,
                     where=(test_residuals < 0), alpha=0.3, color='red',   label='Under-predicted')
axes[0].set_title('Test Residuals Over Time')
axes[0].set_ylabel('Residual (USD)')
axes[0].set_xlabel('Test Sample Index')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residual distribution
axes[1].hist(test_residuals, bins=40, color='steelblue', edgecolor='black', alpha=0.8)
axes[1].axvline(0,                       color='red',    linestyle='--', linewidth=1.5, label='Zero')
axes[1].axvline(test_residuals.mean(),   color='orange', linestyle='--', linewidth=1.5,
                label=f'Mean: ${test_residuals.mean():.2f}')
axes[1].set_title('Residual Distribution')
axes[1].set_xlabel('Residual (USD)')
axes[1].set_ylabel('Frequency')
axes[1].legend()

# Predicted vs Actual scatter
axes[2].scatter(prices_test, pred_test_prices, alpha=0.4, s=10, color='steelblue')
_mn, _mx = prices_test.min(), prices_test.max()
axes[2].plot([_mn, _mx], [_mn, _mx], 'r--', linewidth=1.5, label='Perfect prediction')
axes[2].set_title('Predicted vs Actual')
axes[2].set_xlabel('Actual Price ($)')
axes[2].set_ylabel('Predicted Price ($)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('QLSTM — Test Set Residual Analysis', fontsize=14)
plt.tight_layout()
plt.savefig("qlstm_residuals.png", dpi=150)
plt.show()


#  12. Error by Market Regime
#  Regimes defined by 20-day rolling return:
#  Bull  = top tercile, Bear = bottom tercile, Sideways = middle
test_prices_s  = pd.Series(prices_test.flatten())
test_roll_ret  = test_prices_s.pct_change(20)   # 20-day momentum

tercile_lo = test_roll_ret.quantile(0.33)
tercile_hi = test_roll_ret.quantile(0.67)

regime = pd.cut(test_roll_ret,
                bins=[-np.inf, tercile_lo, tercile_hi, np.inf],
                labels=['Bear', 'Sideways', 'Bull'])

abs_errors = np.abs(test_residuals)
regime_df  = pd.DataFrame({'regime': regime, 'abs_error': abs_errors}).dropna()

regime_mae  = regime_df.groupby('regime', observed=True)['abs_error'].mean()
regime_cnt  = regime_df.groupby('regime', observed=True)['abs_error'].count()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = {'Bear': '#e74c3c', 'Sideways': "#c7ab7f", 'Bull': '#2ecc71'}
regime_mae.plot(kind='bar', ax=axes[0],
                color=[colors[r] for r in regime_mae.index],
                edgecolor='black', alpha=0.85)
axes[0].set_title('Mean Absolute Error by Market Regime')
axes[0].set_ylabel('MAE (USD)')
axes[0].set_xlabel('Regime')
axes[0].tick_params(axis='x', rotation=0)
for i, (v, n) in enumerate(zip(regime_mae, regime_cnt)):
    axes[0].text(i, v + 0.3, f'${v:.2f}\n(n={n})', ha='center', fontsize=9)
axes[0].grid(True, alpha=0.3, axis='y')

regime_df.boxplot(column='abs_error', by='regime', ax=axes[1],
                  patch_artist=True)
axes[1].set_title('Error Distribution by Regime')
axes[1].set_xlabel('Regime')
axes[1].set_ylabel('Absolute Error (USD)')
plt.sca(axes[1])
plt.xticks(rotation=0)

plt.suptitle('QLSTM — Error by Market Regime', fontsize=14)
plt.tight_layout()
plt.savefig("qlstm_regime_analysis.png", dpi=150)
plt.show()

print('\nRegime MAE Summary:')
print(pd.DataFrame({'MAE ($)': regime_mae.round(2), 'N samples': regime_cnt}).to_string())


#  13. Rolling Performance Window (30-day)
ROLL_WIN = 30

roll_mae  = pd.Series(abs_errors).rolling(ROLL_WIN).mean()
roll_mape = pd.Series(
    np.abs(test_residuals / prices_test.flatten()) * 100
).rolling(ROLL_WIN).mean()

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(roll_mae,  color='steelblue', linewidth=1.5, label=f'{ROLL_WIN}-day rolling MAE')
axes[0].axhline(abs_errors.mean(), color='red', linestyle='--', linewidth=1,
                label=f'Overall MAE: ${abs_errors.mean():.2f}')
axes[0].set_ylabel('MAE (USD)')
axes[0].set_title(f'Rolling {ROLL_WIN}-Day MAE — Test Set')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(roll_mape, color='darkorange', linewidth=1.5, label=f'{ROLL_WIN}-day rolling MAPE')
axes[1].axhline(roll_mape.mean(), color='red', linestyle='--', linewidth=1,
                label=f'Mean MAPE: {roll_mape.mean():.2f}%')
axes[1].set_ylabel('MAPE (%)')
axes[1].set_xlabel('Test Sample Index')
axes[1].set_title(f'Rolling {ROLL_WIN}-Day MAPE — Test Set')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('QLSTM — Rolling Performance Over Time', fontsize=14)
plt.tight_layout()
plt.savefig("qlstm_rolling_performance.png", dpi=150)
plt.show()

print("Done!")