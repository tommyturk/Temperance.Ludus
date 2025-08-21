import argparse
import json
import sys
import traceback
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Calculate simplye moving average (SMA), standard deviation, and RSI
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

# Inherit strategy class from 'Strategy' in backtesting.py library.

class MeanReversionRsi(Strategy):
    # default settings for the strategy.
    # moving average period, RSI parameters, standard deviation multiplier, ATR settings, and max holding bars.
    ma_period = 20
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    std_dev_multiplier = 2.0
    atr_period = 14
    atr_multiplier = 2.5
    max_holding_bars = 10

    # The 'init' method is called once at the beginning of the backtest.
    # It calculates all the technical indicators that the strategy will use.
    def init(self):
        close = self.data.Close
        # self.I() tells the library to calculate and manage this indicator
        # self.sma is the simple moving average of the closing prices.
        self.sma = self.I(calculate_sma, close, self.ma_period)
        # self.std is the standard deviation of the closing prices.
        self.std = self.I(calculate_std, close, self.ma_period)
        # These lines create the upper and lower BB
        # They are lines above and below the moving average that show how far the price is from the average.
        self.upper_band = self.sma + self.std * self.std_dev_multiplier
        self.lower_band = self.sma - self.std * self.std_dev_multiplier
        # self.rsi is  relative strength index, which tells us if the asset is overbought or oversold.
        self.rsi = self.I(calculate_rsi, close, self.rsi_period)
        
        high = self.data.High
        low = self.data.Low
        # 'tr' is the 'True Range' used for ATR.
        tr = pd.DataFrame({'High': high, 'Low': low, 'PrevClose': pd.Series(close).shift()})
        # true_range calculate the largest of three values to measure price volatility:
        true_range = tr.apply(lambda row: max(row.High - row.Low, abs(row.High - row.PrevClose), abs(row.Low - row.PrevClose)), axis=1)
        # self.are is the ATR which measures market volatility.
        self.atr = self.I(calculate_sma, true_range.to_numpy(), self.atr_period)

    # The 'next' method is the core of the strategy. 
    # Its called on every new bar (or time period, like a 15-minunte candle).
    def next(self):
        # This block checks if we are currenty in a trade (self.position)
        if self.position:
            # 1. Profit Target at Mean
            if self.position.is_long and crossover(self.data.Close, self.sma):
                self.position.close()
                return
            elif self.position.is_short and crossover(self.sma, self.data.Close):
                self.position.close()
                return

            # 2. Time Stop
            if len(self.trades) > 0 and (len(self.data) - self.trades[-1].entry_bar) >= self.max_holding_bars:
                self.position.close()
                return

        # --- Entry Logic (No changes here) ---
        # Only checked if we are not in a trade.
        if not self.position:
            # This is the long entry (buying)
            # checks two conditions:
            # RSI is below 30 'rsi_oversold', meaning the stock is potentially oversold.
            # The price has crossed below the lower BB suggesting a potentioal rebound.
            if self.rsi[-1] < self.rsi_oversold and crossover(self.data.Close, self.lower_band):
                # This calculates the stop-loss price based on the current price and the ATR, accounts for volatility.
                stop_loss = self.data.Close[-1] - self.atr[-1] * self.atr_multiplier
                # self.buy() places a buy order with a set stop-loss.
                self.buy(sl=stop_loss)
            # This is the **short entry
            # It checks if the RSI is above 70 'rsi_overbought', meaning the stock is potentially overbought. Crossing above the upper BB suggests a potential price drop.
            elif self.rsi[-1] > self.rsi_overbought and crossover(self.upper_band, self.data.Close):
                stop_loss = self.data.Close[-1] + self.atr[-1] * self.atr_multiplier
                self.sell(sl=stop_loss)

# ... (rest of the file is unchanged) ...
# The code below handles the execution of the script from the command line.
# It sets up the backtesting environment, runs the optimization, and reports the results.
def main():
    try:
        # This sets up the program to accept inputs from the command line, like the stock symbol and data file path.
        parser = argparse.ArgumentParser(description="Ludus Backtest-Driven Optimizer")
        parser.add_argument("--symbol", type=str, required=True)
        parser.add_argument("--interval", type=str, required=True)
        parser.add_argument("--data_path", type=str, required=True)
        args = parser.parse_args()

        print(f"--- Starting Optimization for {args.symbol} [{args.interval}] ---")

        # This reads the historical data from the CSV file provided.
        dataframe = pd.read_csv(args.data_path)
        # It converts the 'Timestamp' column to a proper date/time format.
        dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp'])
        #It renames the columns to match what the backtesting.py library expects.
        dataframe.set_index('Timestamp', inplace=True)
        dataframe.rename(columns={'OpenPrice': 'Open', 'HighPrice': 'High', 'LowPrice': 'Low', 'ClosePrice': 'Close', 'Volume': 'Volume'}, inplace=True)
        
        # This is a basic check to make sure you have enough data to run the backtest.
        if len(dataframe) < 100:
             raise ValueError(f"Insufficient data: got {len(dataframe)} rows")

        # This initializes the 'Backtest' object.
        # It sets the data, the strategy to use, the starting cash, and the trading commission.
        bt = Backtest(dataframe, MeanReversionRsi, cash=100_000, commission=.002)

        #This is the key part: it runs the optimization.
        # It will test every combination of the settings provided in the ranges specified.
        stats = bt.optimize(
            ma_period=range(20, 101, 20),
            rsi_period=range(10, 31, 5),
            rsi_oversold=[25, 30, 35],
            rsi_overbought=[65, 70, 75],
            std_dev_multiplier=[2.0, 2.5],
            max_holding_bars=[10, 20, 30],
            # 'maximize' tells the optimizer which performance metric to aim for.
            # The Sharpe Ration measures risk-adjusted return.
            maximize='Sharpe Ratio',
            # 'constraint' is a rule to ensure the settings are logicial (e.g. the oversold value must be lower than the overbought value).
            constraint=lambda p: p.rsi_oversold < p.rsi_overbought
        )

        print("--- Optimization Complete ---")
        # 'stats' holds all the results, includinc the best-performing parameters.
        best_params = stats._strategy._params

        # This creates a dictionary ('result') to hold all the important information in a structured way.
        result = {
            "Status": "Completed",
            "OptimizedParameters": {
                # extract the best found parameters from the stats object.
                "MovingAveragePeriod": int(best_params['ma_period']),
                "RSIPeriod": int(best_params['rsi_period']),
                "RSIOversold": float(best_params['rsi_oversold']),
                "RSIOverbought": float(best_params['rsi_overbought']),
                "StdDevMultiplier": float(best_params['std_dev_multiplier']),
                "MaxHoldingBars": int(best_params['max_holding_bars'])
            },
            "Performance": {
                # extract the performance metrics of the best strategy.
                "SharpeRatio": stats['Sharpe Ratio'], "ReturnPct": stats['Return [%]'],
                "WinRatePct": stats['Win Rate [%]'], "TotalTrades": stats['# Trades']
            }
        }
        
        print(json.dumps(result))

    except Exception as e:
        error_details = {"Error": str(e), "Traceback": traceback.format_exc()}
        print(f"PYTHON_ERROR: {json.dumps(error_details)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()