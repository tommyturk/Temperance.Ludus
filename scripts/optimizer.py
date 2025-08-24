# In scripts/optimizer.py

import argparse
import json
import numpy as np
import pandas as pd
import sys
import traceback
import time
import vectorbt as vbt
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import optuna

# --- GPU Configuration ---
warnings.filterwarnings('ignore')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- Vectorbt Settings ---
vbt.settings.array_wrapper = 'cupy'
vbt.settings.returns['year_freq'] = '365 days'
vbt.settings.portfolio['init_cash'] = 100000.0

# --- Helper Functions ---
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_model(trial, time_steps, n_features, n_outputs):
    model = Sequential()
    model.add(LSTM(trial.suggest_int('units_1', 50, 200),
                   return_sequences=trial.suggest_categorical('return_seq', [True, False]),
                   input_shape=(time_steps, n_features)))
    if trial.suggest_categorical('return_seq', [True, False]):
        model.add(LSTM(trial.suggest_int('units_2', 50, 200)))
    model.add(Dropout(trial.suggest_float('dropout', 0.1, 0.5)))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)),
                  loss='mean_squared_error')
    return model

def objective(trial, time_steps, n_features, n_outputs, X_train, y_train, X_val, y_val):
    model = build_model(trial, time_steps, n_features, n_outputs)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=50, batch_size=trial.suggest_int('batch_size', 32, 128),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                        verbose=0)
    return min(history.history['val_loss'])


def main():
    parser = argparse.ArgumentParser(description="Ludus LSTM-Enhanced Optimizer")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=['train', 'finetune'], required=True)
    parser.add_argument("--model_path", type=str, default="/app/models/lstm_optimizer.h5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--ma_period_range", type=int, nargs=3, required=True)
    parser.add_argument("--rsi_period_range", type=int, nargs=3, required=True)
    parser.add_argument("--rsi_oversold_range", type=int, nargs=3, required=True)
    parser.add_argument("--rsi_overbought_range", type=int, nargs=3, required=True)
    parser.add_argument("--std_dev_multiplier_range", type=float, nargs=3, required=True)
    parser.add_argument("--atr_period_range", type=int, nargs=3, required=True)
    parser.add_argument("--atr_multiplier_range", type=float, nargs=3, required=True)
    args = parser.parse_args()

    try:
        overall_start_time = time.time()
        print("--- [DIAG] Starting LSTM-Enhanced Optimization ---")

        price_df = pd.read_csv(args.data_path, index_col='Timestamp', parse_dates=True)
        price = price_df['ClosePrice']
        high = price_df['HighPrice']
        low = price_df['LowPrice']

        print("--- [DIAG] Stage 1: Running brute-force parameter search with vectorbt ---")
        ma_periods = np.arange(*args.ma_period_range)
        std_multipliers = np.arange(*args.std_dev_multiplier_range)
        rsi_periods = np.arange(*args.rsi_period_range)
        rsi_oversold = np.arange(*args.rsi_oversold_range)
        rsi_overbought = np.arange(*args.rsi_overbought_range)
        atr_periods = np.arange(*args.atr_period_range)
        atr_multipliers = np.arange(*args.atr_multiplier_range)

        # --- THE DEFINITIVE FIX: Manual Bollinger Band Calculation ---
        # This robustly calculates the bands and avoids the library's internal bugs.
        ma_output = vbt.MA.run(price, window=ma_periods, short_name='ma')
        stdev_output = price.vbt.rolling_std(window=ma_periods, ddof=1)

        # Broadcast them together manually to create the bands
        upper_band = ma_output.ma + (stdev_output * std_multipliers)
        lower_band = ma_output.ma - (stdev_output * std_multipliers)
        # --- END OF FIX ---

        rsi = vbt.RSI.run(price, window=rsi_periods, short_name='rsi').rsi
        atr = vbt.ATR.run(high, low, price, window=atr_periods, short_name='atr').atr

        entries = (price.vbt.crossed_below(lower_band)) & (rsi < rsi_oversold)
        exits = (price.vbt.crossed_above(upper_band)) & (rsi > rsi_overbought)
        sl_stop = atr * atr_multipliers

        portfolio = vbt.Portfolio.from_signals(price, entries=entries, exits=exits, sl_stop=sl_stop, freq='1D')

        total_trades = portfolio.total_trades()
        valid_mask = total_trades > 5
        if not valid_mask.any():
            raise ValueError("No parameter combination resulted in more than 5 trades during brute-force search.")

        sharpe_ratio = portfolio.sharpe_ratio()
        best_params_idx = sharpe_ratio[valid_mask].idxmax()

        # Extract best parameters by their names from the multi-level index
        best_ma = best_params_idx[best_params_idx.index.get_level_values('ma_window').name]
        best_std_mult = best_params_idx[best_params_idx.index.get_level_values('rolling_std_multiplier').name]
        best_rsi_period = best_params_idx[best_params_idx.index.get_level_values('rsi_window').name]
        best_rsi_os = best_params_idx[best_params_idx.index.get_level_values('RSI_less').name]
        best_rsi_ob = best_params_idx[best_params_idx.index.get_level_values('RSI_greater').name]
        best_atr_period = best_params_idx[best_params_idx.index.get_level_values('atr_window').name]
        best_atr_mult = best_params_idx[best_params_idx.index.get_level_values('ATR_multiplier').name]

        brute_force_results = {
            "MovingAveragePeriod": int(best_ma),
            "StdDevMultiplier": float(best_std_mult),
            "RSIPeriod": int(best_rsi_period),
            "RSIOversold": int(best_rsi_os),
            "RSIOverbought": int(best_rsi_ob),
            "AtrPeriod": int(best_atr_period),
            "AtrMultiplier": float(best_atr_mult)
        }

        print(f"--- [DIAG] Brute-force best parameters found: {brute_force_results} ---")
        
        # --- Stage 2 (LSTM) and beyond remains the same ---
        print(f"--- [DIAG] Stage 2: Preparing data for LSTM model ({args.mode} mode) ---")
        features = pd.DataFrame(index=price_df.index)
        features['returns'] = price_df['ClosePrice'].pct_change()
        features['volatility'] = features['returns'].rolling(window=21).std()
        features['momentum'] = price_df['ClosePrice'] / price_df['ClosePrice'].rolling(window=21).mean()
        features.dropna(inplace=True)

        targets = pd.DataFrame([brute_force_results] * len(features), index=features.index)

        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = feature_scaler.fit_transform(features)
        scaled_targets = target_scaler.fit_transform(targets)
        scaled_features_df = pd.DataFrame(scaled_features, index=features.index, columns=features.columns)

        X, y = create_dataset(scaled_features_df, pd.DataFrame(scaled_targets, index=targets.index), time_steps=args.lookback)

        if args.mode == 'train':
            print("--- [DIAG] Training new LSTM model with Optuna hyperparameter search ---")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, args.lookback, X.shape[2], y.shape[1], X_train, y_train, X_val, y_val), n_trials=20)
            print(f"--- [DIAG] Best hyperparameters found: {study.best_params} ---")
            model = build_model(study.best_trial, args.lookback, X.shape[2], y.shape[1])
            checkpoint = ModelCheckpoint(args.model_path, save_best_only=True, monitor='val_loss', mode='min')
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=args.epochs, batch_size=study.best_params['batch_size'],
                      callbacks=[early_stopping, checkpoint], verbose=1)
        elif args.mode == 'finetune':
            print(f"--- [DIAG] Fine-tuning existing model from {args.model_path} ---")
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Model file not found at {args.model_path}. Run in 'train' mode first.")
            model = load_model(args.model_path)
            model.fit(X, y, epochs=args.epochs, batch_size=32, verbose=1)
            model.save(args.model_path)

        print("--- [DIAG] Predicting parameters for the next trading day ---")
        last_lookback_features = scaled_features_df.iloc[-args.lookback:].values
        next_day_input = np.reshape(last_lookback_features, (1, args.lookback, last_lookback_features.shape[1]))
        predicted_scaled_params = model.predict(next_day_input)
        predicted_params = target_scaler.inverse_transform(predicted_scaled_params)[0]

        final_parameters = {
            "MovingAveragePeriod": int(round(predicted_params[0])),
            "StdDevMultiplier": float(round(predicted_params[1], 2)),
            "RSIPeriod": int(round(predicted_params[2])),
            "RSIOversold": int(round(predicted_params[3])),
            "RSIOverbought": int(round(predicted_params[4])),
            "AtrPeriod": int(round(predicted_params[5])),
            "AtrMultiplier": float(round(predicted_params[6], 2))
        }

        result = {
            "Status": "Completed",
            "Mode": args.mode,
            "OptimizedParameters": final_parameters,
            "BruteForcePerformance": {
                "SharpeRatio": float(portfolio.sharpe_ratio()[best_params_idx]),
                "TotalReturn": float(portfolio.total_return()[best_params_idx]),
                "TotalTrades": int(portfolio.total_trades()[best_params_idx]),
                "WinRatePct": float(portfolio.win_rate()[best_params_idx] * 100),
                "MaxDrawdownPct": float(portfolio.max_drawdown()[best_params_idx] * 100)
            }
        }

        print(f"--- [DIAG] Overall process complete in {time.time() - overall_start_time:.2f} seconds. ---")
        print("--- Optimization Complete ---")
        print(json.dumps(result))

    except Exception as e:
        error_details = {"Error": str(e), "Traceback": traceback.format_exc()}
        print(f"PYTHON_ERROR: {json.dumps(error_details)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()