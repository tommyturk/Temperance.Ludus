import argparse
import json
import sys
import traceback
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def calculate_sma(array, n):
    return pd.Series(array).rolling(window=n, min_periods=n).mean()

def calculate_std(array, n):
    return pd.Series(array).rolling(window=n, min_periods=n).std()

def calculate_rsi(array, n):
    delta = pd.Series(array).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

class MeanReversionRsi(Strategy):
    ma_period = 20
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    std_dev_multiplier = 2.0
    atr_period = 14
    atr_multiplier = 2.5
    max_holding_bars = 10

    def init(self):
        close = self.data.Close
        self.sma = self.I(calculate_sma, close, self.ma_period)
        self.std = self.I(calculate_std, close, self.ma_period)
        self.upper_band = self.sma + self.std * self.std_dev_multiplier
        self.lower_band = self.sma - self.std * self.std_dev_multiplier
        self.rsi = self.I(calculate_rsi, close, self.rsi_period)
        
        # ATR requires high, low, close for True Range calculation
        high = self.data.High
        low = self.data.Low
        tr = pd.DataFrame({'High': high, 'Low': low, 'PrevClose': pd.Series(close).shift()})
        true_range = tr.apply(lambda row: max(row.High - row.Low, abs(row.High - row.PrevClose), abs(row.Low - row.PrevClose)), axis=1)
        self.atr = self.I(calculate_sma, true_range.to_numpy(), self.atr_period)


    def next(self):
        # --- Multi-Layered Exit Logic ---
        if self.position:
            # 1. ATR Stop-Loss
            if self.position.is_long and self.data.Low[-1] <= self.position.sl:
                self.position.close()
                return
            elif self.position.is_short and self.data.High[-1] >= self.position.sl:
                self.position.close()
                return
            
            # 2. Profit Target at Mean
            if self.position.is_long and crossover(self.data.Close, self.sma):
                self.position.close()
                return
            elif self.position.is_short and crossover(self.sma, self.data.Close):
                self.position.close()
                return

            # 3. Time Stop
            if len(self.trades) > 0 and (len(self.data) - self.trades[-1].entry_bar) >= self.max_holding_bars:
                self.position.close()
                return

        # --- Entry Logic ---
        if not self.position:
            if self.rsi[-1] < self.rsi_oversold and crossover(self.data.Close, self.lower_band):
                stop_loss = self.data.Close[-1] - self.atr[-1] * self.atr_multiplier
                self.buy(sl=stop_loss)
            elif self.rsi[-1] > self.rsi_overbought and crossover(self.upper_band, self.data.Close):
                stop_loss = self.data.Close[-1] + self.atr[-1] * self.atr_multiplier
                self.sell(sl=stop_loss)

def main():
    try:
        parser = argparse.ArgumentParser(description="Ludus Backtest-Driven Optimizer")
        parser.add_argument("--symbol", type=str, required=True)
        parser.add_argument("--interval", type=str, required=True)
        parser.add_argument("--data_path", type=str, required=True)
        args = parser.parse_args()

        print(f"--- Starting Optimization for {args.symbol} [{args.interval}] ---")

        dataframe = pd.read_csv(args.data_path)
        dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp'])
        dataframe.set_index('Timestamp', inplace=True)
        dataframe.rename(columns={'OpenPrice': 'Open', 'HighPrice': 'High', 'LowPrice': 'Low', 'ClosePrice': 'Close', 'Volume': 'Volume'}, inplace=True)
        
        if len(dataframe) < 100:
             raise ValueError(f"Insufficient data: got {len(dataframe)} rows")

        bt = Backtest(dataframe, MeanReversionRsi, cash=100_000, commission=.002)

        stats = bt.optimize(
            ma_period=range(20, 101, 20),
            rsi_period=range(10, 31, 5),
            rsi_oversold=[25, 30, 35],
            rsi_overbought=[65, 70, 75],
            std_dev_multiplier=[2.0, 2.5],
            max_holding_bars=[10, 20, 30],
            maximize='Sharpe Ratio',
            constraint=lambda p: p.rsi_oversold < p.rsi_overbought
        )

        print("--- Optimization Complete ---")
        best_params = stats._strategy._params

        result = {
            "Status": "Completed",
            "OptimizedParameters": {
                "MovingAveragePeriod": int(best_params['ma_period']),
                "RSIPeriod": int(best_params['rsi_period']),
                "RSIOversold": float(best_params['rsi_oversold']),
                "RSIOverbought": float(best_params['rsi_overbought']),
                "StdDevMultiplier": float(best_params['std_dev_multiplier']),
                "MaxHoldingBars": int(best_params['max_holding_bars'])
            },
            "Performance": {
                "SharpeRatio": stats['Sharpe Ratio'], "ReturnPct": stats['Return [%]'],
                "WinRatePct": stats['Win Rate [%]'], "TotalTrades": stats['# Trades']
            }
        }
        
        # The final JSON output is printed to stdout for the C# service to capture
        print(json.dumps(result))

    except Exception as e:
        # Error details are printed to stderr
        error_details = {"Error": str(e), "Traceback": traceback.format_exc()}
        print(f"PYTHON_ERROR: {json.dumps(error_details)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()