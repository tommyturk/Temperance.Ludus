# In scripts/optimizer.py

import argparse
import json
import numpy as np
import pandas as pd
import sys
import traceback
import time
import vectorbt as vbt

# --- This is the correct syntax to enable the GPU backend ---
vbt.settings['array_wrapper'] = 'cupy'

def main():
    parser = argparse.ArgumentParser(description="Ludus GPU-Accelerated Vectorized Optimizer")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ma_period_range", type=int, nargs=3, required=True)
    parser.add_argument("--rsi_period_range", type=int, nargs=3, required=True)
    parser.add_argument("--rsi_oversold_range", type=int, nargs=3, required=True)
    parser.add_argument("--rsi_overbought_range", type=int, nargs=3, required=True)
    parser.add_argument("--std_dev_multiplier_range", type=float, nargs=3, required=True)
    parser.add_argument("--atr_period_range", type=int, nargs=3, required=True)
    parser.add_argument("--atr_multiplier_range", type=float, nargs=3, required=True)

    args = parser.parse_args()

    try:
        # --- [DIAG] GPU Detection ---
        import cupy as cp
        try:
            gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name']
            print(f"--- [DIAG] GPU Detected and Initialized: {gpu_name.decode('utf-8')} ---")
        except Exception as e:
            print(f"--- [DIAG] GPU NOT DETECTED. Error: {e} ---")
            sys.exit(1) # Exit if GPU is not found

        overall_start_time = time.time()
        print("--- [DIAG] Starting Optimization ---")
        
        # --- Corrected Data Loading using pandas ---
        start_time = time.time()
        df = pd.read_csv(args.data_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        price = df['ClosePrice']
        high = df['HighPrice']
        low = df['LowPrice']
        print(f"--- [DIAG] Data Loading complete in {time.time() - start_time:.2f} seconds. Shape: {price.shape} ---")

        # --- Generate parameter combinations ---
        ma_periods = np.arange(*args.ma_period_range)
        rsi_periods = np.arange(*args.rsi_period_range)
        rsi_oversold_levels = np.arange(*args.rsi_oversold_range)
        rsi_overbought_levels = np.arange(*args.rsi_overbought_range)
        std_multipliers = np.arange(*args.std_dev_multiplier_range)
        atr_periods = np.arange(*args.atr_period_range)
        atr_multipliers = np.arange(*args.atr_multiplier_range)
        
        # --- Indicator Calculation (Purely using vectorbt's GPU-native functions) ---
        start_time = time.time()
        ma = vbt.MA.run(price, window=ma_periods, short_name='ma')
        std = price.vbt.rolling_std(window=ma_periods)
        
        upper_band = ma.ma + std.rolling_std * std_multipliers
        lower_band = ma.ma - std.rolling_std * std_multipliers

        rsi = vbt.RSI.run(price, window=rsi_periods, short_name='rsi')
        atr = vbt.ATR.run(high, low, price, window=atr_periods, short_name='atr')
        
        long_entries = price.vbt.crossed_below(lower_band) & (rsi.rsi < rsi_oversold_levels)
        short_entries = price.vbt.crossed_above(upper_band) & (rsi.rsi > rsi_overbought_levels)
        
        long_sl_stop = atr.atr * atr_multipliers
        print(f"--- [DIAG] Indicator calculation complete in {time.time() - start_time:.2f} seconds. ---")

        # --- Portfolio Backtesting (Purely on GPU) ---
        start_time = time.time()
        portfolio = vbt.Portfolio.from_signals(
            price,
            entries=long_entries,
            exits=short_entries,
            sl_stop=long_sl_stop,
            freq='1h' 
        )
        print(f"--- [DIAG] Portfolio backtesting complete in {time.time() - start_time:.2f} seconds. ---")

        # --- Finding Best Parameters ---
        start_time = time.time()
        valid_sharpe = portfolio.sharpe_ratio()[portfolio.total_trades() > 0]
        if valid_sharpe.empty:
            raise ValueError("No trades were executed for any parameter combination.")

        best_params_idx = valid_sharpe.idxmax()
        best_performance = valid_sharpe.max()
        
        (best_ma, best_std_mult, best_rsi, best_rsi_os, best_rsi_ob, best_atr, best_atr_mult) = best_params_idx
        print(f"--- [DIAG] Best parameter search complete in {time.time() - start_time:.2f} seconds. ---")

        result = {
            "Status": "Completed",
            "OptimizedParameters": {
                "MovingAveragePeriod": int(best_ma), "StdDevMultiplier": float(best_std_mult),
                "RSIPeriod": int(best_rsi), "RSIOversold": int(best_rsi_os),
                "RSIOverbought": int(best_rsi_ob), "AtrPeriod": int(best_atr),
                "AtrMultiplier": float(best_atr_mult)
            },
            "Performance": {
                "SharpeRatio": float(best_performance), "TotalReturn": float(portfolio.total_return()[best_params_idx]),
                "TotalTrades": int(portfolio.total_trades()[best_params_idx]), "WinRatePct": float(portfolio.win_rate()[best_params_idx] * 100),
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