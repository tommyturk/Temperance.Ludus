import sys
import json
import pandas as pd
import cudf
import cupy as cp
import numpy as np
import itertools
import argparse
import os
from pynvml import *

# Suppress TensorFlow logging and initialize GPU memory growth
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# --- Configuration ---
LABEL_GENERATION_PERIOD = 252 # How many periods to use for each brute-force backtest to generate a label

# --- Helper & Backtesting Functions ---
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def calculate_rsi_gpu(prices: cudf.Series, period: int) -> cudf.Series:
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50)

def vectorized_backtest_gpu(prices: cudf.Series, entries: cp.ndarray, exits: cp.ndarray) -> float:
    prices_cp = cp.asarray(prices)
    signals = cp.zeros_like(prices_cp, dtype=cp.int8)
    signals[entries] = 1
    signals[exits] = -1
    positions = cp.asarray(cudf.Series(signals).replace(to_replace=-1, method='ffill'))
    positions[positions == -1] = 0
    positions_shifted = cp.roll(positions, 1)
    positions_shifted[0] = 0
    log_returns = cp.log(prices_cp / cp.roll(prices_cp, 1))
    log_returns[0] = 0
    strategy_returns = positions_shifted * log_returns
    total_return = cp.exp(cp.sum(strategy_returns)) - 1.0
    return float(total_return)

# --- Machine Learning Functions ---
def create_model(input_shape, output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_shape, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def get_features_gpu(prices_gdf: cudf.DataFrame):
    close = prices_gdf['ClosePrice']
    features = cudf.DataFrame(index=prices_gdf.index)
    features['rsi_14'] = calculate_rsi_gpu(close, 14) / 100.0
    features['rsi_28'] = calculate_rsi_gpu(close, 28) / 100.0
    rolling_mean_20 = close.rolling(20).mean()
    rolling_std_20 = close.rolling(20).std()
    features['bb_width'] = ((rolling_mean_20 + 2 * rolling_std_20) - (rolling_mean_20 - 2 * rolling_std_20)) / rolling_mean_20
    features['return_5d'] = close.pct_change(5)
    return features.fillna(0).astype(np.float32)

def generate_training_labels(prices_gdf, param_grid, model_lookback_window):
    features_gdf = get_features_gpu(prices_gdf)
    close_prices = prices_gdf['ClosePrice']
    X_train, y_train = [], []
    num_steps = len(prices_gdf) - LABEL_GENERATION_PERIOD - model_lookback_window
    if num_steps <= 0: return np.array([]), np.array([])

    param_combinations = list(itertools.product(*param_grid.values()))
    for i in range(num_steps):
        if i % 100 == 0: eprint(f"Label generation progress: {i}/{num_steps}")
        start_idx = i + model_lookback_window
        end_idx = start_idx + LABEL_GENERATION_PERIOD
        window_prices = close_prices.iloc[start_idx:end_idx]
        best_perf, best_p_values = -float('inf'), None
        for params in param_combinations:
            p = dict(zip(param_grid.keys(), params))
            mean = window_prices.rolling(window=p['ma_period']).mean()
            std = window_prices.rolling(window=p['ma_period']).std()
            entries = (cp.asarray(window_prices) < cp.asarray(mean - std * p['std_dev_multiplier'])) & (cp.asarray(calculate_rsi_gpu(window_prices, p['rsi_period'])) < p['rsi_oversold'])
            exits = cp.asarray(window_prices) > cp.asarray(mean)
            perf = vectorized_backtest_gpu(window_prices, entries, exits)
            if perf > best_perf:
                best_perf, best_p_values = perf, list(p.values())
        if best_p_values:
            X_train.append(features_gdf.iloc[i:i + model_lookback_window].to_numpy().flatten())
            y_train.append(best_p_values)
    return np.array(X_train), np.array(y_train)

# --- Main Logic ---
def main(args):
    prices_gdf = cudf.from_pandas(pd.read_csv(args.input_csv_path, parse_dates=['Timestamp']).set_index('Timestamp'))
    param_grid = {
        'ma_period': range(15, 31, 5), 'std_dev_multiplier': np.arange(1.8, 2.5, 0.2),
        'rsi_period': range(10, 21, 5), 'rsi_oversold': range(25, 36, 5), 'rsi_overbought': range(65, 76, 5)
    }
    input_shape = args.lookback * len(get_features_gpu(prices_gdf.head(args.lookback + 5)).columns)
    output_shape = len(param_grid)

    if args.mode == 'train':
        X_train, y_train = generate_training_labels(prices_gdf, param_grid, args.lookback)
        if len(X_train) == 0: raise ValueError("Failed to generate any training labels.")
        model = create_model(input_shape, output_shape)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)
        model.save(args.model_path)
    elif args.mode == 'fine-tune':
        if not os.path.exists(args.model_path): raise FileNotFoundError(f"Model not found at {args.model_path}")
        model = tf.keras.models.load_model(args.model_path)
        X_tune, y_tune = generate_training_labels(prices_gdf, param_grid, args.lookback)
        if len(X_tune) > 0:
            model.fit(X_tune, y_tune, epochs=args.epochs, batch_size=4, verbose=0)
            model.save(args.model_path)

    model = tf.keras.models.load_model(args.model_path)
    features_gdf = get_features_gpu(prices_gdf)
    last_window = features_gdf.tail(args.lookback).to_numpy().flatten().reshape(1, -1)
    predicted_params_raw = model.predict(last_window, verbose=0)[0]
    final_params = {
        "MovingAveragePeriod": int(np.round(np.clip(predicted_params_raw[0], min(param_grid['ma_period']), max(param_grid['ma_period'])))),
        "StdDevMultiplier": float(np.round(np.clip(predicted_params_raw[1], min(param_grid['std_dev_multiplier']), max(param_grid['std_dev_multiplier'])), 2)),
        "RSIPeriod": int(np.round(np.clip(predicted_params_raw[2], min(param_grid['rsi_period']), max(param_grid['rsi_period'])))),
        "RSIOversold": int(np.round(np.clip(predicted_params_raw[3], min(param_grid['rsi_oversold']), max(param_grid['rsi_oversold'])))),
        "RSIOverbought": int(np.round(np.clip(predicted_params_raw[4], min(param_grid['rsi_overbought']), max(param_grid['rsi_overbought']))))
    }
    with open(args.output_json_path, 'w') as f:
        json.dump({"status": "success", "best_parameters": final_params}, f, indent=4)
    eprint(f"Predicted parameters written to {args.output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv_path"); parser.add_argument("output_json_path")
    parser.add_argument("--mode", required=True, choices=['train', 'fine-tune'])
    parser.add_argument("--model-path", required=True); parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lookback", type=int, default=60)
    try:
        main(parser.parse_args())
    except Exception as e:
        output_path = sys.argv[2] if len(sys.argv) > 2 else "error_output.json"
        with open(output_path, 'w') as f: json.dump({"status": "error", "message": str(e)}, f, indent=4)
        sys.exit(1)