# In scripts/optimizer.py

import argparse
import json
import numpy as np
import pandas as pd
import sys
import traceback
import time
import vectorbt as vbt

def custom_strategy_logic(close, high, low, ma_window, std_mult, rsi_window, rsi_oversold, rsi_overbought, atr_window, atr_mult, **kwargs):
    """The complete, self-contained strategy logic."""
    # --- THIS IS THE CRITICAL FIX ---
    # Flatten the 2D numpy arrays from vectorbt into 1D arrays for pandas
    close_pd = pd.Series(close.flatten())
    high_pd = pd.Series(high.flatten())
    low_pd = pd.Series(low.flatten())
    
    # Bollinger Bands
    ma = close_pd.rolling(int(ma_window), min_periods=int(ma_window)).mean()
    std = close_pd.rolling(int(ma_window), min_periods=int(ma_window)).std()
    upper_band = ma + std * std_mult
    lower_band = ma - std * std_mult
    
    # RSI
    rsi = vbt.RSI.run(close_pd, window=int(rsi_window)).rsi
    
    # ATR
    atr = vbt.ATR.run(high_pd, low_pd, close_pd, window=int(atr_window)).atr

    # Signals
    long_entries = (close_pd < lower_band) & (rsi < rsi_oversold)
    short_entries = (close_pd > upper_band) & (rsi > rsi_overbought)
    
    sl_stop = atr * atr_mult
    
    # Return numpy arrays as expected by vectorbt
    return long_entries.values, short_entries.values, sl_stop.values

def main():
    parser = argparse.ArgumentParser(description="Ludus GPU-Accelerated Vectorized Optimizer with ATR Stops")
    parser.add_argument("--data_path", type=str, required=True)
    
    # Parameter ranges
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
        print("--- [DIAG] Starting GPU-Accelerated Optimization ---")
        
        # --- [DIAG] Step 1: Data Loading ---
        start_time = time.time()
        df = pd.read_csv(args.data_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        print(f"--- [DIAG] Data Loading complete in {time.time() - start_time:.2f} seconds. Shape: {df.shape} ---")
        
        price = df['ClosePrice']
        high = df['HighPrice']
        low = df['LowPrice']

        # --- [DIAG] Step 2: Parameter Generation ---
        start_time = time.time()
        ma_periods = np.arange(*args.ma_period_range)
        rsi_periods = np.arange(*args.rsi_period_range)
        rsi_oversold_levels = np.arange(*args.rsi_oversold_range)
        rsi_overbought_levels = np.arange(*args.rsi_overbought_range)
        std_multipliers = np.arange(*args.std_dev_multiplier_range)
        atr_periods = np.arange(*args.atr_period_range)
        atr_multipliers = np.arange(*args.atr_multiplier_range)
        
        total_combinations = len(ma_periods) * len(std_multipliers) * len(rsi_periods) * len(rsi_oversold_levels) * len(rsi_overbought_levels) * len(atr_periods) * len(atr_multipliers)
        print(f"--- [DIAG] Parameter generation complete in {time.time() - start_time:.2f} seconds. Total combinations to test: {total_combinations:,} ---")

        # --- [DIAG] Step 3: Indicator Calculation ---
        start_time = time.time()
        MeanReversionStrategy = vbt.IndicatorFactory(
            class_name='MeanReversionStrategy',
            input_names=['close', 'high', 'low'],
            param_names=['ma_window', 'std_mult', 'rsi_window', 'rsi_oversold', 'rsi_overbought', 'atr_window', 'atr_mult'],
            output_names=['long_entries', 'short_entries', 'sl_stop']
        ).from_apply_func(custom_strategy_logic, takes_1d=True)

        strat = MeanReversionStrategy.run(
            price, high, low,
            ma_window=ma_periods,
            std_mult=std_multipliers,
            rsi_window=rsi_periods,
            rsi_oversold=rsi_oversold_levels,
            rsi_overbought=rsi_overbought_levels,
            atr_window=atr_periods,
            atr_mult=atr_multipliers,
            param_product=True
        )
        print(f"--- [DIAG] Indicator calculation complete in {time.time() - start_time:.2f} seconds. ---")

        # --- [DIAG] Step 4: Portfolio Backtesting ---
        start_time = time.time()
        portfolio = vbt.Portfolio.from_signals(
            price,
            entries=strat.long_entries,
            exits=strat.short_entries,
            sl_stop=strat.sl_stop,
            freq='1h' 
        )
        print(f"--- [DIAG] Portfolio backtesting complete in {time.time() - start_time:.2f} seconds. ---")

        # --- [DIAG] Step 5: Finding Best Parameters ---
        start_time = time.time()
        valid_sharpe = portfolio.sharpe_ratio()[portfolio.total_trades() > 0]
        if valid_sharpe.empty:
            raise ValueError("No trades were executed for any parameter combination.")

        best_params_idx = valid_sharpe.idxmax()
        best_performance = valid_sharpe.max()
        
        (best_ma, best_std, best_rsi, best_oversold, best_overbought, best_atr, best_atr_mult) = best_params_idx
        print(f"--- [DIAG] Best parameter search complete in {time.time() - start_time:.2f} seconds. ---")

        result = {
            "Status": "Completed",
            "OptimizedParameters": {
                "MovingAveragePeriod": int(best_ma), "StdDevMultiplier": float(best_std),
                "RSIPeriod": int(best_rsi), "RSIOversold": float(best_oversold),
                "RSIOverbought": float(best_overbought), "AtrPeriod": int(best_atr),
                "AtrMultiplier": float(best_atr_mult)
            },
            "Performance": {
                "SharpeRatio": best_performance, "TotalReturn": portfolio.total_return()[best_params_idx],
                "TotalTrades": portfolio.total_trades()[best_params_idx], "WinRatePct": portfolio.win_rate()[best_params_idx] * 100,
                "MaxDrawdownPct": portfolio.max_drawdown()[best_params_idx] * 100
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