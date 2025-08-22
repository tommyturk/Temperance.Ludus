# In scripts/optimizer.py

import argparse
import json
import numpy as np
import pandas as pd
import sys
import traceback
import time
import vectorbt as vbt

# --- THE DEFINITIVE FIX: Use the correct API to set the GPU wrapper ---
# The 'array_wrapper' key must be set within the 'wrapping' dictionary.
vbt.settings.wrapping['array_wrapper'] = 'cupy'
vbt.settings.returns['year_freq'] = '365 days'
vbt.settings.portfolio['init_cash'] = 100000.0

def main():
    parser = argparse.ArgumentParser(description="Ludus vectorbt GPU Optimizer")
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
        overall_start_time = time.time()
        print("--- [DIAG] Starting vectorbt GPU Optimization ---")

        # --- Data Loading (vectorbt handles GPU transfer automatically) ---
        start_time = time.time()
        # Using the exact column names from your HistoricalPriceModel.cs
        price = vbt.CSVData.from_files(args.data_path).get('ClosePrice')
        high = vbt.CSVData.from_files(args.data_path).get('HighPrice')
        low = vbt.CSVData.from_files(args.data_path).get('LowPrice')
        print(f"--- [DIAG] Data Loading complete in {time.time() - start_time:.2f} seconds ---")

        # --- Generate Parameter Ranges ---
        ma_periods = np.arange(*args.ma_period_range)
        std_multipliers = np.arange(*args.std_dev_multiplier_range)
        rsi_periods = np.arange(*args.rsi_period_range)
        rsi_oversold = np.arange(*args.rsi_oversold_range)
        rsi_overbought = np.arange(*args.rsi_overbought_range)
        atr_periods = np.arange(*args.atr_period_range)
        atr_multipliers = np.arange(*args.atr_multiplier_range)

        # --- Massively Parallel Indicator Calculation on GPU ---
        total_combinations = len(ma_periods) * len(std_multipliers) * len(rsi_periods) * len(rsi_oversold) * len(rsi_overbought) * len(atr_periods) * len(atr_multipliers)
        print(f"--- [DIAG] Calculating indicators for {total_combinations:,} combinations ---")
        
        ma = vbt.MA.run(price, window=ma_periods, short_name='ma')
        std_dev = vbt.talib('STDDEV').run(price, timeperiod=ma_periods).real
        
        upper_band = ma.ma_crossed + std_dev * std_multipliers
        lower_band = ma.ma_crossed - std_dev * std_multipliers

        rsi = vbt.RSI.run(price, window=rsi_periods, short_name='rsi')
        atr = vbt.ATR.run(high, low, price, window=atr_periods, short_name='atr')

        # --- Vectorized Signal Generation on GPU ---
        entries = (price.vbt.crossed_below(lower_band)) & (rsi.rsi_crossed < rsi_oversold)
        exits = (price.vbt.crossed_above(upper_band)) & (rsi.rsi_crossed > rsi_overbought)
        
        sl_stop = atr.atr_crossed * atr_multipliers

        # --- Fully Vectorized Portfolio Backtest on GPU ---
        portfolio = vbt.Portfolio.from_signals(
            price,
            entries=entries,
            exits=exits,
            sl_stop=sl_stop,
            freq='1D' # Assuming daily data for annualization
        )
        
        # --- Find Best Parameters ---
        total_trades = portfolio.total_trades()
        valid_mask = total_trades > 5 
        
        if not valid_mask.any():
            raise ValueError("No parameter combination resulted in more than 5 trades.")

        sharpe_ratio = portfolio.sharpe_ratio()
        best_sharpe = sharpe_ratio[valid_mask].max()
        best_params_idx = sharpe_ratio[valid_mask].idxmax()
        
        (best_ma, best_std_mult, best_rsi, best_rsi_os, best_rsi_ob, best_atr, best_atr_mult) = best_params_idx

        result = {
            "Status": "Completed",
            "OptimizedParameters": {
                "MovingAveragePeriod": int(best_ma),
                "StdDevMultiplier": float(best_std_mult),
                "RSIPeriod": int(best_rsi),
                "RSIOversold": int(best_rsi_os),
                "RSIOverbought": int(best_rsi_ob),
                "AtrPeriod": int(best_atr),
                "AtrMultiplier": float(best_atr_mult)
            },
            "Performance": {
                "SharpeRatio": float(best_sharpe),
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