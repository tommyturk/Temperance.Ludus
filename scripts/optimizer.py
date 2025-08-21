# In scripts/optimizer.py

import argparse
import json
import numpy as np
import pandas as pd
import sys
import traceback
import vectorbt as vbt

def main():
    parser = argparse.ArgumentParser(description="Ludus GPU-Accelerated Vectorized Optimizer with ATR Stops")
    parser.add_argument("--data_path", type=str, required=True)
    
    # --- Parameter ranges passed from C# ---
    parser.add_argument("--ma_period_range", type=int, nargs=3, required=True)
    parser.add_argument("--rsi_period_range", type=int, nargs=3, required=True)
    parser.add_argument("--rsi_oversold_range", type=int, nargs=3, required=True)
    parser.add_argument("--rsi_overbought_range", type=int, nargs=3, required=True)
    parser.add_argument("--std_dev_multiplier_range", type=float, nargs=3, required=True)
    parser.add_argument("--atr_period_range", type=int, nargs=3, required=True)
    parser.add_argument("--atr_multiplier_range", type=float, nargs=3, required=True)

    args = parser.parse_args()

    try:
        print("--- Starting GPU-Accelerated Optimization with ATR Volatility Exits ---")
        
        # Load and prepare data
        df = pd.read_csv(args.data_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        
        price = df['ClosePrice']
        high = df['HighPrice']
        low = df['LowPrice']

        # --- Generate parameter combinations ---
        ma_periods = np.arange(*args.ma_period_range)
        rsi_periods = np.arange(*args.rsi_period_range)
        rsi_oversold_levels = np.arange(*args.rsi_oversold_range)
        rsi_overbought_levels = np.arange(*args.rsi_overbought_range)
        std_multipliers = np.arange(*args.std_dev_multiplier_range)
        atr_periods = np.arange(*args.atr_period_range)
        atr_multipliers = np.arange(*args.atr_multiplier_range)

        # --- THIS IS THE CORRECTED SECTION: Define a robust, self-contained strategy indicator ---
        @vbt.cached_indicator
        def MeanReversionStrategy(close, high, low, ma_window, std_mult, rsi_window, rsi_oversold, rsi_overbought, atr_window, atr_mult):
            # Bollinger Bands
            ma = vbt.MA.run(close, ma_window).ma
            std = vbt.MA.run(close, ma_window).std
            upper_band = ma + std * std_mult
            lower_band = ma - std * std_mult
            
            # RSI
            rsi = vbt.RSI.run(close, rsi_window).rsi
            
            # ATR
            atr = vbt.ATR.run(high, low, close, atr_window).atr

            # Signals
            long_entries = (close < lower_band) & (rsi < rsi_oversold)
            short_entries = (close > upper_band) & (rsi > rsi_overbought)
            
            sl_stop = atr * atr_mult
            
            return long_entries, short_entries, sl_stop

        # Create a single factory for the entire strategy
        strat_indicator = MeanReversionStrategy.run(
            price, high, low,
            ma_window=ma_periods,
            std_mult=std_multipliers,
            rsi_window=rsi_periods,
            rsi_oversold=rsi_oversold_levels,
            rsi_overbought=rsi_overbought_levels,
            atr_window=atr_periods,
            atr_mult=atr_multipliers,
            param_product=True  # This creates the full grid
        )

        # --- Run the Backtest ---
        print(f"Running vectorized backtest...")
        portfolio = vbt.Portfolio.from_signals(
            price,
            entries=strat_indicator.long_entries,
            exits=strat_indicator.short_entries,
            sl_stop=strat_indicator.sl_stop,
            freq='1h' 
        )

        # --- Find the Best Performing Combination ---
        valid_sharpe = portfolio.sharpe_ratio()[portfolio.total_trades() > 0]
        if valid_sharpe.empty:
            raise ValueError("No trades were executed for any parameter combination.")

        best_params_idx = valid_sharpe.idxmax()
        best_performance = valid_sharpe.max()
        
        (best_ma, best_std, best_rsi, best_oversold, best_overbought, best_atr, best_atr_mult) = best_params_idx

        result = {
            "Status": "Completed",
            "OptimizedParameters": {
                "MovingAveragePeriod": int(best_ma),
                "StdDevMultiplier": float(best_std),
                "RSIPeriod": int(best_rsi),
                "RSIOversold": float(best_oversold),
                "RSIOverbought": float(best_overbought),
                "AtrPeriod": int(best_atr),
                "AtrMultiplier": float(best_atr_mult)
            },
            "Performance": {
                "SharpeRatio": best_performance,
                "TotalReturn": portfolio.total_return()[best_params_idx],
                "TotalTrades": portfolio.total_trades()[best_params_idx],
                "WinRatePct": portfolio.win_rate()[best_params_idx] * 100,
                "MaxDrawdownPct": portfolio.max_drawdown()[best_params_idx] * 100
            }
        }
        
        print("--- Optimization Complete ---")
        print(json.dumps(result))

    except Exception as e:
        error_details = {"Error": str(e), "Traceback": traceback.format_exc()}
        print(f"PYTHON_ERROR: {json.dumps(error_details)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()