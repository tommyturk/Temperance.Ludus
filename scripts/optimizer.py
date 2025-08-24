import sys
import json
import pandas as pd
import cudf
import cupy as cp
import numpy as np
import itertools
from pynvml import *

# --- Helper Functions ---

def eprint(*args, **kwargs):
    """Prints to stderr, which is captured by the .NET process for logging."""
    print(*args, file=sys.stderr, **kwargs)

def check_gpu():
    """Checks for GPU presence and prints info."""
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        if device_count == 0:
            eprint("Error: No NVIDIA GPU detected by NVML.")
            return False
        handle = nvmlDeviceGetHandleByIndex(0)
        eprint(f"GPU Found: {nvmlDeviceGetName(handle)}")
        nvmlShutdown()
        return True
    except NVMLError as error:
        eprint(f"Error initializing NVML: {error}. Is the NVIDIA driver installed?")
        return False

def calculate_rsi_gpu(prices: cudf.Series, period: int) -> cudf.Series:
    """Calculates RSI entirely on the GPU using cuDF and CuPy."""
    delta = prices.diff()
    
    gains = delta.copy()
    gains[gains < 0] = 0
    
    losses = delta.copy()
    losses[losses > 0] = 0
    losses = losses.abs()

    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    # Use cupy for the element-wise formula
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Fill initial NaNs that result from the rolling window
    rsi = rsi.fillna(50) # A common practice is to set initial RSI to a neutral 50
    return rsi

def vectorized_backtest_gpu(prices: cudf.Series, entries: cp.ndarray, exits: cp.ndarray) -> float:
    """
    Performs a simple, vectorized backtest on the GPU to calculate total log return.
    """
    # Ensure inputs are cupy arrays for GPU operations
    prices_cp = cp.asarray(prices)
    
    # Generate a positions array: 1 for long, 0 for flat
    positions = cp.zeros_like(prices_cp, dtype=cp.int8)
    
    # Determine when we are in a trade
    in_trade = cp.zeros_like(prices_cp, dtype=bool)
    is_holding = False
    for i in range(len(prices_cp)):
        if not is_holding and entries[i]:
            is_holding = True
        elif is_holding and exits[i]:
            is_holding = False
        in_trade[i] = is_holding

    positions[in_trade] = 1
    
    # Shift positions by 1 to align trades with the correct return period
    positions_shifted = cp.roll(positions, 1)
    positions_shifted[0] = 0 # Start with no position

    # Calculate log returns on the GPU
    log_returns = cp.log(prices_cp / cp.roll(prices_cp, 1))
    log_returns[0] = 0 # First return is always zero

    # Calculate strategy returns
    strategy_returns = positions_shifted * log_returns
    
    # Calculate total return
    total_return = cp.exp(cp.sum(strategy_returns)) - 1.0
    
    return float(total_return) # Return as a standard Python float

# --- Main Execution Logic ---

def main():
    """Main function to run the optimization."""
    if not check_gpu():
        sys.exit(1)

    # --- 1. Argument Parsing ---
    if len(sys.argv) != 3:
        eprint(f"Usage: python {sys.argv[0]} <input_csv_path> <output_json_path>")
        sys.exit(1)

    input_csv_path = sys.argv[1]
    output_json_path = sys.argv[2]
    eprint(f"Starting optimization. Input: {input_csv_path}, Output: {output_json_path}")

    # --- 2. Data Loading (Pandas -> cuDF) ---
    try:
        # Load with pandas first for robust CSV parsing
        prices_pdf = pd.read_csv(input_csv_path, parse_dates=['Timestamp'])
        prices_pdf = prices_pdf.sort_values(by='Timestamp').set_index('Timestamp')
        
        # Move data to GPU
        prices_gdf = cudf.from_pandas(prices_pdf)
        close_prices = prices_gdf['ClosePrice']
        eprint(f"Successfully loaded {len(prices_gdf)} data points onto GPU.")

    except Exception as e:
        eprint(f"Error loading data: {e}")
        sys.exit(1)

    # --- 3. Define Parameter Grid ---
    param_grid = {
        'ma_period': range(15, 31, 5),          # e.g., [15, 20, 25, 30]
        'std_dev_multiplier': np.arange(1.8, 2.5, 0.2), # e.g., [1.8, 2.0, 2.2, 2.4]
        'rsi_period': range(10, 21, 5),          # e.g., [10, 15, 20]
        'rsi_oversold': range(25, 36, 5),        # e.g., [25, 30, 35]
        'rsi_overbought': range(65, 76, 5)       # e.g., [65, 70, 75]
    }
    
    param_combinations = list(itertools.product(*param_grid.values()))
    eprint(f"Generated {len(param_combinations)} parameter combinations to test.")

    # --- 4. Optimization Loop ---
    best_performance = -float('inf')
    best_params = None

    for i, params in enumerate(param_combinations):
        p = dict(zip(param_grid.keys(), params))
        
        # Calculate Indicators on GPU
        rolling_mean = close_prices.rolling(window=p['ma_period']).mean()
        rolling_std = close_prices.rolling(window=p['ma_period']).std()
        upper_band = rolling_mean + (rolling_std * p['std_dev_multiplier'])
        lower_band = rolling_mean - (rolling_std * p['std_dev_multiplier'])
        rsi = calculate_rsi_gpu(close_prices, p['rsi_period'])

        # Generate Signals on GPU (returns CuPy arrays)
        entries = (cp.asarray(close_prices) < cp.asarray(lower_band)) & (cp.asarray(rsi) < p['rsi_oversold'])
        exits = (cp.asarray(close_prices) > cp.asarray(rolling_mean)) # Exit when price reverts to the mean

        # Run Backtest
        performance = vectorized_backtest_gpu(close_prices, entries, exits)

        if performance > best_performance:
            best_performance = performance
            best_params = p
            eprint(f"New best params found (Perf: {performance:.4f}): {p}")

    # --- 5. Output Results ---
    if best_params:
        result = {
            "status": "success",
            "best_parameters": {
                # Ensure values are standard Python types for JSON serialization
                "MovingAveragePeriod": int(best_params['ma_period']),
                "StdDevMultiplier": float(best_params['std_dev_multiplier']),
                "RSIPeriod": int(best_params['rsi_period']),
                "RSIOversold": int(best_params['rsi_oversold']),
                "RSIOverbought": int(best_params['rsi_overbought'])
            },
            "performance": best_performance
        }
        eprint(f"Optimization complete. Best performance: {best_performance:.4f}")
    else:
        result = {"status": "error", "message": "No profitable parameters found."}
        eprint("Optimization failed to find any profitable parameters.")

    with open(output_json_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    eprint(f"Results saved to {output_json_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        eprint(f"An unhandled exception occurred: {e}")
        # Create an error JSON for the output file
        error_output = {
            "status": "error",
            "message": str(e)
        }
        # Try to write error to output file if path is available
        if len(sys.argv) == 3:
            with open(sys.argv[2], 'w') as f:
                json.dump(error_output, f, indent=4)
        sys.exit(1)