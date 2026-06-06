# LSTM for prediction XAU/USD
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers


# Reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()


# 1. Load data
df = pd.read_csv(r"XAU_1d_data.csv", sep=";")
df = df[['Date', 'Open', 'High', 'Low', 'Close']]
df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
df = df.set_index('Date').sort_index()

# 2. Feature engineering
df['Return']   = np.log(df['Close'] / df['Close'].shift(1))  # daily log-return
df['HL_range'] = np.log(df['High']  / df['Low'])             # daily volatility (High-Low spread)
df['OC_ret']   = np.log(df['Close'] / df['Open'])            # intraday direction (Open-Close)
df['MA5_ret']  = df['Return'].rolling(5).mean()              # short-term momentum
df['MA20_ret'] = df['Return'].rolling(20).mean()             # medium-term momentum
df.dropna(inplace=True)

#  External features: DXY (USD strength), 10Y Treasury yield, VIX (fear index)
FEATURE_COLS = ['Close', 'Return', 'HL_range', 'OC_ret', 'MA5_ret', 'MA20_ret']

_start = df.index.min().strftime('%Y-%m-%d')
_end   = (df.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
try:
    dxy = yf.download('DX-Y.NYB', start=_start, end=_end, auto_adjust=True, progress=False)['Close']
    tnx = yf.download('^TNX',     start=_start, end=_end, auto_adjust=True, progress=False)['Close']
    vix = yf.download('^VIX',     start=_start, end=_end, auto_adjust=True, progress=False)['Close']
    for series in (dxy, tnx, vix):
        series.index = pd.to_datetime(series.index).normalize()
    df['DXY_ret']    = np.log(dxy / dxy.shift(1)).reindex(df.index).ffill()
    df['TNX_change'] = tnx.diff().reindex(df.index).ffill()          # yield change, not log-return
    df['VIX_ret']    = np.log(vix / vix.shift(1)).reindex(df.index).ffill()
    df.dropna(subset=['DXY_ret', 'TNX_change', 'VIX_ret'], inplace=True)
    FEATURE_COLS += ['DXY_ret', 'TNX_change', 'VIX_ret']
    print(f"External features loaded: DXY, TNX, VIX ({len(df)} rows remaining)")
except Exception as e:
    print(f"External feature download failed ({e}) — continuing without DXY/TNX/VIX")
N_FEATURES   = len(FEATURE_COLS)
N_WINDOW     = 60



def make_windows(dataframe, n, feature_cols):
    # Slide a window over dataframe using integer indexing
    data       = dataframe[feature_cols].values.astype(np.float32)
    return_col = feature_cols.index('Return')
    X          = np.stack([data[i : i + n] for i in range(len(data) - n)])  # (T, n, F)
    y          = data[n:, return_col]                                         # (T,)  next-day log-return
    dates      = dataframe.index[n:]
    return X, y.astype(np.float32), dates

X, y, dates = make_windows(df, N_WINDOW, FEATURE_COLS)

# 4. Splitting data — 80% train, 10% validation, 10% test
q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)

X_train, y_train, dates_train = X[:q_80],        y[:q_80],        dates[:q_80]
X_val,   y_val,   dates_val   = X[q_80:q_90],    y[q_80:q_90],    dates[q_80:q_90]
X_test,  y_test,  dates_test  = X[q_90:],        y[q_90:],        dates[q_90:]

# Plot last Close in each window to visualise the split boundaries
plt.plot(dates_train, X_train[:, -1, 0])
plt.plot(dates_val,   X_val[:, -1, 0])
plt.plot(dates_test,  X_test[:, -1, 0])
plt.legend(['Train', 'Validation', 'Test'])
plt.title('Close Price — split')
plt.show()

# 5. Scale after splitting 
scalers     = [MinMaxScaler() for _ in FEATURE_COLS]

X_train_sc  = X_train.copy()
X_val_sc    = X_val.copy()
X_test_sc   = X_test.copy()

for i, scaler in enumerate(scalers):
    scaler.fit(X_train[:, :, i].reshape(-1, 1))
    X_train_sc[:, :, i] = scaler.transform(X_train[:, :, i].reshape(-1, 1)).reshape(X_train.shape[0], N_WINDOW)
    X_val_sc[:, :, i]   = scaler.transform(X_val[:, :, i].reshape(-1, 1)).reshape(X_val.shape[0],   N_WINDOW)
    X_test_sc[:, :, i]  = scaler.transform(X_test[:, :, i].reshape(-1, 1)).reshape(X_test.shape[0], N_WINDOW)

# Scale the return target with StandardScaler (returns are symmetric around 0, not bounded, so MinMaxScaler is a poor fit here)

return_scaler = StandardScaler()
return_scaler.fit(y_train.reshape(-1, 1))
y_train_sc = return_scaler.transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
y_val_sc   = return_scaler.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)
y_test_sc  = return_scaler.transform(y_test.reshape(-1, 1)).ravel().astype(np.float32)

# 6. Building LSTM model — Dropout included between LSTM layers to reduce overfitting
model = Sequential([
    layers.Input((N_WINDOW, N_FEATURES)),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])

# MSE loss
model.compile(loss='mse',
              optimizer=Adam(learning_rate=0.001),
              metrics=['mae'])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)


#Save per-epoch train/validation MAE from a Keras History object to CSV.
def save_mae_history_to_csv(history, output_path='train_val_mae.csv'):
    history_dict = history.history
    train_mae = history_dict.get('mae')
    val_mae = history_dict.get('val_mae')

    if train_mae is None or val_mae is None:
        raise ValueError("Could not find 'mae' or 'val_mae' in training history.")

    mae_df = pd.DataFrame({
        'epoch': np.arange(1, len(train_mae) + 1),
        'train_mae': train_mae,
        'val_mae': val_mae,
    })
    mae_df.to_csv(output_path, index=False)
    print(f"MAE history saved to {output_path}")

history = model.fit(X_train_sc, y_train_sc,
                    validation_data=(X_val_sc, y_val_sc),
                    epochs=100,
                    callbacks=[early_stop])

save_mae_history_to_csv(history, output_path='train_val_mae.csv')

# 7. Predict returns, then reconstruct USD prices
#    predicted_price = last_close_in_window × exp(predicted_log_return)
def returns_to_prices(raw_model_output, last_closes):
    predicted_returns = return_scaler.inverse_transform(raw_model_output.reshape(-1, 1))
    return last_closes.reshape(-1, 1) * np.exp(predicted_returns)

train_predictions = returns_to_prices(model.predict(X_train_sc), X_train[:, -1, 0])
val_predictions   = returns_to_prices(model.predict(X_val_sc),   X_val[:, -1, 0])
test_predictions  = returns_to_prices(model.predict(X_test_sc),  X_test[:, -1, 0])

# Actual next-day price = last_close × exp(actual_log_return)
y_train_actual = (X_train[:, -1, 0] * np.exp(y_train)).reshape(-1, 1)
y_val_actual   = (X_val[:, -1, 0]   * np.exp(y_val)).reshape(-1, 1)
y_test_actual  = (X_test[:, -1, 0]  * np.exp(y_test)).reshape(-1, 1)

# 8. Plots
plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train_actual)
plt.legend(['Training Predictions', 'Training Observations'])
plt.title('Training Set')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.show()

plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val_actual)
plt.legend(['Validation Predictions', 'Validation Observations'])
plt.title('Validation Set')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.show()

plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test_actual)
plt.legend(['Testing Predictions', 'Testing Observations'])
plt.title('Test Set')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.show()

plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train_actual)
plt.plot(dates_val,   val_predictions)
plt.plot(dates_val,   y_val_actual)
plt.plot(dates_test,  test_predictions)
plt.plot(dates_test,  y_test_actual)
plt.legend(['Training Predictions', 'Training Observations',
            'Validation Predictions', 'Validation Observations',
            'Testing Predictions', 'Testing Observations'])
plt.title('Train / Validation / Test')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.show()

# 9. Evaluation metrics
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
    if len(actual_dir) > 0:
        directional_accuracy = np.mean(np.sign(actual_dir) == np.sign(pred_dir)) * 100
    else:
        directional_accuracy = float('nan')

    return {
        'Set':                  set_name,
        'MSE (USD²)':           f'{mse:.4f}',
        'RMSE (USD)':           f'{rmse:.4f}',
        'MAE (USD)':            f'{mae:.4f}',
        'MAPE (%)':             f'{mape:.2f}',
        'R² Score':             f'{r2:.4f}',
        'Directional Acc (%)':  f'{directional_accuracy:.2f}',
    }

train_metrics = compute_metrics(y_train_actual, train_predictions, 'Train')
val_metrics   = compute_metrics(y_val_actual,   val_predictions,   'Validation')
test_metrics  = compute_metrics(y_test_actual,  test_predictions,  'Test')

metrics_df = pd.DataFrame([train_metrics, val_metrics, test_metrics]).set_index('Set')
print('\n' + '='*70)
print('                     MODEL EVALUATION METRICS')
print('='*70)
print(metrics_df.to_string())
print('='*70)

# 10. Residual analysis
test_residuals = y_test_actual.flatten() - test_predictions.flatten()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(dates_test, test_residuals, color='coral')
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[0].set_title('Test Set Residuals Over Time')
axes[0].set_ylabel('Residual (USD)')
axes[0].set_xlabel('Date')

axes[1].hist(test_residuals, bins=30, color='steelblue', edgecolor='black')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1)
axes[1].set_title('Test Residual Distribution')
axes[1].set_xlabel('Residual (USD)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

model.save('xauusd_lstm_final.h5')
print("Model saved successfully")
