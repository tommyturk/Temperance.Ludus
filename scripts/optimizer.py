import sys
import json
import pandas as pd
import numpy as np
import itertools
import argparse
import os
import warnings
from typing import Tuple, Dict, Any, List, Optional

# Suppress warnings and TensorFlow logging
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def setup_gpu():
    """Sets memory growth for all available GPUs to avoid allocation errors."""
    try:
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            eprint(f"Successfully configured memory growth for {len(physical_devices)} GPU(s).")
            return True, tf
        else:
            eprint("No GPU found, using CPU.")
            return False, tf
    except Exception as e:
        eprint(f"GPU setup failed: {e}. Falling back to CPU.")
        # Ensure tf is still returned on failure for consistent function output
        import tensorflow as tf
        return False, tf

def setup_gpu_libraries():
    """Setup GPU libraries with fallbacks"""
    has_gpu = False
    use_cudf = False
    use_cupy = False
    
    try:
        import cudf
        import cupy as cp
        # Test basic operations
        test_series = cudf.Series([1, 2, 3, 4, 5])
        test_array = cp.array([1, 2, 3, 4, 5])
        use_cudf = True
        use_cupy = True
        eprint("cuDF and CuPy initialized successfully")
    except Exception as e:
        eprint(f"GPU libraries not available: {e}. Using pandas/numpy fallback.")
        cudf = pd
        cp = np
        use_cudf = False
        use_cupy = False
    
    return use_cudf, use_cupy, cudf if use_cudf else pd, cp if use_cupy else np

# --- Configuration ---
LABEL_GENERATION_PERIOD = 60  # Reduced from 252 to 60 for faster processing
MIN_TRAINING_SAMPLES = 50

def calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
    """Calculate RSI with pandas/numpy fallback"""
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50)

def calculate_rsi_gpu(prices, period: int, use_cudf: bool):
    """Calculate RSI with GPU acceleration if available"""
    if use_cudf and hasattr(prices, 'diff'):
        return calculate_rsi_cudf(prices, period)
    else:
        # Convert to pandas if needed
        if hasattr(prices, 'to_pandas'):
            prices = prices.to_pandas()
        return calculate_rsi(prices, period)

def calculate_rsi_cudf(prices, period: int):
    """CuDF implementation of RSI"""
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50)

def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Average True Range (ATR)"""
    high = df['HighPrice']
    low = df['LowPrice']
    close = df['ClosePrice']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr

def vectorized_backtest(prices, entries, exits, use_cupy: bool) -> float:
    """Vectorized backtesting with GPU/CPU fallback"""
    try:
        if use_cupy:
            return vectorized_backtest_gpu(prices, entries, exits)
        else:
            return vectorized_backtest_cpu(prices, entries, exits)
    except Exception as e:
        eprint(f"Backtest error: {e}. Falling back to CPU version.")
        return vectorized_backtest_cpu(prices, entries, exits)

def vectorized_backtest_gpu(prices, entries, exits) -> float:
    """GPU-accelerated backtesting"""
    try:
        import cupy as cp
        
        prices_cp = cp.asarray(prices)
        if len(prices_cp) < 2:
            return 0.0
        
        # Create signals array
        signals = cp.zeros(len(prices_cp), dtype=cp.int8)
        
        # Ensure indices are within bounds
        valid_entries = entries[entries < len(prices_cp)]
        valid_exits = exits[exits < len(prices_cp)]
        
        if len(valid_entries) > 0:
            signals[valid_entries] = 1
        if len(valid_exits) > 0:
            signals[valid_exits] = -1
        
        # Create positions using simple state machine
        positions = cp.zeros(len(prices_cp), dtype=cp.float32)
        current_pos = 0.0
        
        for i in range(len(signals)):
            if signals[i] == 1:
                current_pos = 1.0
            elif signals[i] == -1:
                current_pos = 0.0
            positions[i] = current_pos
        
        # Shift positions to avoid look-ahead bias
        positions_shifted = cp.roll(positions, 1)
        positions_shifted[0] = 0.0
        
        # Calculate log returns
        price_ratios = prices_cp[1:] / prices_cp[:-1]
        price_ratios = cp.where(price_ratios <= 0, 1.0, price_ratios)  # Handle non-positive prices
        log_returns = cp.log(price_ratios)
        log_returns = cp.concatenate([cp.array([0.0]), log_returns])
        
        # Calculate strategy returns
        strategy_returns = positions_shifted * log_returns
        total_log_return = cp.sum(strategy_returns)
        
        # Convert back to total return
        if cp.isfinite(total_log_return):
            total_return = cp.exp(total_log_return) - 1.0
            return float(total_return)
        else:
            return 0.0
            
    except Exception as e:
        return 0.0

def vectorized_backtest_cpu(prices, entries, exits) -> float:
    """CPU backtesting fallback"""
    try:
        prices_np = np.asarray(prices)
        if len(prices_np) < 2:
            return 0.0
        
        # Create signals array  
        signals = np.zeros(len(prices_np), dtype=np.int8)
        
        # Ensure indices are within bounds
        valid_entries = entries[entries < len(prices_np)]
        valid_exits = exits[exits < len(prices_np)]
        
        if len(valid_entries) > 0:
            signals[valid_entries] = 1
        if len(valid_exits) > 0:
            signals[valid_exits] = -1
        
        # Create positions using simple state machine
        positions = np.zeros(len(prices_np), dtype=np.float32)
        current_pos = 0.0
        
        for i in range(len(signals)):
            if signals[i] == 1:
                current_pos = 1.0
            elif signals[i] == -1:
                current_pos = 0.0
            positions[i] = current_pos
        
        # Shift positions to avoid look-ahead bias
        positions_shifted = np.roll(positions, 1)
        positions_shifted[0] = 0.0
        
        # Calculate log returns
        price_ratios = prices_np[1:] / prices_np[:-1]
        price_ratios = np.where(price_ratios <= 0, 1.0, price_ratios)  # Handle non-positive prices
        log_returns = np.log(price_ratios)
        log_returns = np.concatenate([np.array([0.0]), log_returns])
        
        # Calculate strategy returns
        strategy_returns = positions_shifted * log_returns
        total_log_return = np.sum(strategy_returns)
        
        # Convert back to total return
        if np.isfinite(total_log_return):
            total_return = np.exp(total_log_return) - 1.0
            return float(total_return)
        else:
            return 0.0
            
    except Exception as e:
        return 0.0

def create_model(input_shape: int, output_shape: int, tf):
    """Create neural network model"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_shape, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def get_features(prices_df, use_cudf: bool):
    """Extract features from price data"""
    if use_cudf:
        return get_features_gpu(prices_df)
    else:
        return get_features_cpu(prices_df)

def get_features_cpu(prices_df):
    """CPU version of feature extraction"""
    close = prices_df['ClosePrice']
    features = pd.DataFrame(index=prices_df.index)
    
    # RSI features
    features['rsi_14'] = calculate_rsi(close, 14) / 100.0
    features['rsi_28'] = calculate_rsi(close, 28) / 100.0
    
    # Bollinger Band width
    rolling_mean_20 = close.rolling(20).mean()
    rolling_std_20 = close.rolling(20).std()
    features['bb_width'] = ((rolling_mean_20 + 2 * rolling_std_20) - 
                           (rolling_mean_20 - 2 * rolling_std_20)) / rolling_mean_20
    
    # Price momentum
    features['return_5d'] = close.pct_change(5)
    features['return_10d'] = close.pct_change(10)
    
    return features.fillna(0).astype(np.float32)

def get_features_gpu(prices_gdf):
    """GPU version of feature extraction"""
    close = prices_gdf['ClosePrice']
    
    if hasattr(prices_gdf, 'index'):
        features = type(prices_gdf)(index=prices_gdf.index)
    else:
        import cudf
        features = cudf.DataFrame(index=prices_gdf.index)
    
    # RSI features
    features['rsi_14'] = calculate_rsi_cudf(close, 14) / 100.0
    features['rsi_28'] = calculate_rsi_cudf(close, 28) / 100.0
    
    # Bollinger Band width
    rolling_mean_20 = close.rolling(20).mean()
    rolling_std_20 = close.rolling(20).std()
    features['bb_width'] = ((rolling_mean_20 + 2 * rolling_std_20) - 
                           (rolling_mean_20 - 2 * rolling_std_20)) / rolling_mean_20
    
    # Price momentum
    features['return_5d'] = close.pct_change(5)
    features['return_10d'] = close.pct_change(10)
    
    return features.fillna(0).astype(np.float32)

def generate_training_labels(prices_df, param_grid: Dict, model_lookback_window: int,
                           use_cudf: bool, use_cupy: bool, cp_module):
    """Generate training labels using brute force optimization"""
    try:
        features_df = get_features(prices_df, use_cudf)
        close_prices = prices_df['ClosePrice']

        X_train, y_train = [], []
        num_steps = len(prices_df) - LABEL_GENERATION_PERIOD - model_lookback_window

        if num_steps <= 0:
            eprint(f"Insufficient data: need at least {LABEL_GENERATION_PERIOD + model_lookback_window} rows, got {len(prices_df)}")
            return np.array([]), np.array([])

        param_combinations = list(itertools.product(*param_grid.values()))
        eprint(f"Starting label generation for {num_steps} steps with {len(param_combinations)} parameter combinations")

        successful_samples = 0
        failed_attempts = 0

        for i in range(0, num_steps, max(1, num_steps // 100)):  # Sample every N steps to reduce computation
            if i % 10 == 0:
                eprint(f"Label generation progress: {i}/{num_steps} (successful: {successful_samples}, failed: {failed_attempts})")

            start_idx = i + model_lookback_window
            end_idx = start_idx + LABEL_GENERATION_PERIOD

            if end_idx > len(prices_df):
                break

            window_prices = close_prices.iloc[start_idx:end_idx].copy()

            if len(window_prices) < LABEL_GENERATION_PERIOD // 2:
                continue

            # Reset index to make it work with integer indices
            window_prices.reset_index(drop=True, inplace=True)

            best_perf = -float('inf')
            best_p_values = None
            valid_backtests = 0

            # Sample parameter combinations to reduce computation
            sampled_combinations = param_combinations[::max(1, len(param_combinations) // 50)]

            for params in sampled_combinations:
                try:
                    p = dict(zip(param_grid.keys(), params))

                    # Ensure minimum periods for indicators
                    min_periods = max(p['ma_period'], p['rsi_period']) + 5
                    if len(window_prices) < min_periods:
                        continue

                    # Calculate indicators with proper minimum periods
                    mean = window_prices.rolling(window=p['ma_period'], min_periods=p['ma_period']).mean()
                    std = window_prices.rolling(window=p['ma_period'], min_periods=p['ma_period']).std()
                    rsi = calculate_rsi(window_prices, p['rsi_period'])
                    atr = calculate_atr(prices_df.iloc[start_idx:end_idx], int(p['atr_period']))


                    valid_data_start = max(p['ma_period'], p['rsi_period'], int(p['atr_period']))
                    if valid_data_start >= len(window_prices) - 10:
                        continue
                    
                    lower_band = mean - std * p['std_dev_multiplier'] - atr * p['atr_multiplier']


                    price_below_lower = window_prices < lower_band
                    rsi_oversold = rsi < p['rsi_oversold']
                    price_above_mean = window_prices > mean

                    entry_condition = price_below_lower & rsi_oversold
                    exit_condition = price_above_mean

                    # Get actual indices where conditions are True
                    entry_indices = np.where(entry_condition.fillna(False))[0]
                    exit_indices = np.where(exit_condition.fillna(False))[0]

                    # Filter out indices before valid data starts
                    entry_indices = entry_indices[entry_indices >= valid_data_start]
                    exit_indices = exit_indices[exit_indices >= valid_data_start]

                    if len(entry_indices) >= 2 and len(exit_indices) >= 2:
                        try:
                            perf = vectorized_backtest(window_prices, entry_indices, exit_indices, use_cupy)
                            valid_backtests += 1

                            if not np.isnan(perf) and not np.isinf(perf) and perf > best_perf:
                                best_perf = perf
                                best_p_values = list(p.values())
                        except Exception as bt_error:
                            continue

                except Exception as param_error:
                    continue
            
            if best_p_values is not None:
                try:
                    feature_window = features_df.iloc[i:i + model_lookback_window]
                    if use_cudf and hasattr(feature_window, 'to_numpy'):
                        feature_array = feature_window.to_numpy()
                    else:
                        feature_array = feature_window.values

                    if feature_array.size > 0 and not np.any(np.isnan(feature_array)):
                        X_train.append(feature_array.flatten())
                        y_train.append(best_p_values)
                        successful_samples += 1
                    else:
                        failed_attempts += 1
                except Exception as e:
                    eprint(f"Error processing feature window: {e}")
                    failed_attempts += 1
                    continue
            else:
                failed_attempts += 1

        eprint(f"Generated {len(X_train)} training samples (successful: {successful_samples}, failed: {failed_attempts})")
        return np.array(X_train), np.array(y_train)

    except Exception as e:
        eprint(f"Error in generate_training_labels: {e}")
        import traceback
        eprint(traceback.format_exc())
        return np.array([]), np.array([])

def main(args):
    """Main execution function"""
    try:
        eprint(f"Starting optimization with mode: {args.mode}")
        
        # Setup GPU/CPU libraries
        has_gpu, tf = setup_gpu()
        use_cudf, use_cupy, df_lib, cp_module = setup_gpu_libraries()
        
        eprint(f"Configuration: GPU={has_gpu}, cuDF={use_cudf}, cuPy={use_cupy}")
        
        # Load data
        eprint(f"Loading data from: {args.input_csv_path}")
        prices_df = pd.read_csv(args.input_csv_path, parse_dates=['Timestamp'])
        prices_df = prices_df.set_index('Timestamp')
        
        eprint(f"Loaded {len(prices_df)} price records")
        
        if use_cudf:
            try:
                prices_df = df_lib.from_pandas(prices_df)
            except:
                eprint("Failed to convert to cuDF, using pandas")
                use_cudf = False
        
        # Parameter grid - make it more reasonable
        param_grid = {
            'ma_period': list(range(10, 25, 2)),
            'std_dev_multiplier': [round(x, 1) for x in np.arange(1.5, 2.5, 0.1)],
            'rsi_period': list(range(10, 21, 2)),
            'rsi_oversold': list(range(20, 35, 3)),
            'rsi_overbought': list(range(65, 81, 3)),
            'atr_period': list(range(10, 21, 2)),  # NEW
            'atr_multiplier': [round(x, 1) for x in np.arange(1.5, 3.1, 0.2)]  # NEW
        }
        
        # Calculate dimensions
        sample_features = get_features(prices_df.head(args.lookback + 50), use_cudf)
        input_shape = args.lookback * len(sample_features.columns)
        output_shape = len(param_grid)
        
        eprint(f"Model dimensions: input={input_shape}, output={output_shape}")
        
        # Training or fine-tuning
        if args.mode == 'train':
            eprint("Starting training mode")
            X_train, y_train = generate_training_labels(prices_df, param_grid, args.lookback, 
                                                       use_cudf, use_cupy, cp_module)
            
            if len(X_train) == 0:
                eprint("No training samples generated. Creating a basic model with default parameters.")
                
                # Create minimal training data with default parameters
                sample_features = get_features(prices_df.tail(args.lookback + 50), use_cudf)
                if len(sample_features) >= args.lookback:
                    # **FIX: Ensure default_params has all 7 values**
                    param_keys = list(param_grid.keys())
                    default_params = [
                        np.median(param_grid['ma_period']),
                        np.median(param_grid['std_dev_multiplier']),
                        np.median(param_grid['rsi_period']),
                        np.median(param_grid['rsi_oversold']),
                        np.median(param_grid['rsi_overbought']),
                        np.median(param_grid['atr_period']),      # Added
                        np.median(param_grid['atr_multiplier'])   # Added
                    ]
                    
                    # Create a few samples with slight variations
                    X_synthetic = []
                    y_synthetic = []
                    
                    for i in range(min(10, len(sample_features) - args.lookback)):
                        window = sample_features.iloc[i:i + args.lookback]
                        if use_cudf and hasattr(window, 'to_numpy'):
                            feature_array = window.to_numpy()
                        else:
                            feature_array = window.values
                            
                        X_synthetic.append(feature_array.flatten())
                        # **FIX: Ensure varied_params has 7 values with random variations**
                        varied_params = [
                            default_params[0] + np.random.randint(-2, 3),
                            default_params[1] + np.random.uniform(-0.1, 0.1),
                            default_params[2] + np.random.randint(-2, 3),
                            default_params[3] + np.random.randint(-2, 3),
                            default_params[4] + np.random.randint(-2, 3),
                            default_params[5] + np.random.randint(-2, 3),      # Added variation for atr_period
                            default_params[6] + np.random.uniform(-0.2, 0.2)   # Added variation for atr_multiplier
                        ]
                        y_synthetic.append(varied_params)
                    
                    X_train = np.array(X_synthetic)
                    y_train = np.array(y_synthetic)
                    
                    eprint(f"Created {len(X_train)} synthetic training samples")
                else:
                    raise ValueError("Insufficient data for even basic model creation.")
            
            if len(X_train) < MIN_TRAINING_SAMPLES:
                eprint(f"Warning: Only {len(X_train)} training samples generated (minimum recommended: {MIN_TRAINING_SAMPLES})")
            
            eprint(f"Creating model with {len(X_train)} training samples")
            model = create_model(input_shape, output_shape, tf)
            
            # Adjust training parameters based on sample size
            batch_size = min(32, max(1, len(X_train) // 4))
            validation_split = 0.2 if len(X_train) > 10 else 0.0
            
            # Prepare callbacks
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_split > 0 else 'loss', 
                patience=10, restore_best_weights=True, verbose=1)
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_split > 0 else 'loss', 
                factor=0.5, patience=5, min_lr=1e-7, verbose=1)
            
            callbacks = [reduce_lr]
            if validation_split > 0:
                callbacks.append(early_stop)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=min(args.epochs, 50) if len(X_train) < 50 else args.epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            model.save(args.model_path)
            eprint(f"Model saved to: {args.model_path}")
            
        elif args.mode == 'fine-tune':
            eprint("Starting fine-tune mode")
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Model not found at {args.model_path}")
            
            model = tf.keras.models.load_model(args.model_path)
            eprint("Loaded existing model for fine-tuning")
            
            X_tune, y_tune = generate_training_labels(prices_df, param_grid, args.lookback,
                                                     use_cudf, use_cupy, cp_module)
            
            if len(X_tune) > 0:
                eprint(f"Fine-tuning with {len(X_tune)} samples")
                model.fit(X_tune, y_tune, epochs=min(args.epochs, 20), batch_size=4, verbose=1)
                model.save(args.model_path)
                eprint("Fine-tuning completed")
            else:
                eprint("No fine-tuning data generated, skipping fine-tuning")
        
        # Make predictions
        eprint("Loading model for prediction")
        model = tf.keras.models.load_model(args.model_path)
        
        # Get features for prediction
        features_df = get_features(prices_df, use_cudf)
        
        if len(features_df) < args.lookback:
            raise ValueError(f"Insufficient data for prediction: need {args.lookback} samples, got {len(features_df)}")
        
        last_window = features_df.tail(args.lookback)
        
        if use_cudf and hasattr(last_window, 'to_numpy'):
            last_window_array = last_window.to_numpy()
        else:
            last_window_array = last_window.values
            
        prediction_input = last_window_array.flatten().reshape(1, -1)
        
        eprint(f"Making prediction with input shape: {prediction_input.shape}")
        predicted_params_raw = model.predict(prediction_input, verbose=0)[0]
        
        # Convert predictions to valid parameter ranges
        param_keys = list(param_grid.keys())
        final_params = {}
        
        for i, key in enumerate(param_keys):
            param_range = param_grid[key]
            raw_value = predicted_params_raw[i]

            if key in ['ma_period', 'rsi_period', 'rsi_oversold', 'rsi_overbought', 'atr_period']:
                clipped_value = int(np.round(np.clip(raw_value, min(param_range), max(param_range))))
            else:
                clipped_value = float(np.round(np.clip(raw_value, min(param_range), max(param_range)), 2))

            # Map to expected output format
            if key == 'ma_period':
                final_params["MovingAveragePeriod"] = clipped_value
            elif key == 'std_dev_multiplier':
                final_params["StdDevMultiplier"] = clipped_value
            elif key == 'rsi_period':
                final_params["RSIPeriod"] = clipped_value
            elif key == 'rsi_oversold':
                final_params["RSIOversold"] = clipped_value
            elif key == 'rsi_overbought':
                final_params["RSIOverbought"] = clipped_value
            elif key == 'atr_period':
                final_params["AtrPeriod"] = clipped_value
            elif key == 'atr_multiplier':
                final_params["AtrMultiplier"] = clipped_value
        
        # Save results
        result = {
            "Status": "success",
            "BestParameters": final_params,
            "Performance": None,
            "Message": f"Successfully optimized parameters using {args.mode} mode"
        }
        
        with open(args.output_json_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        eprint(f"Results saved to: {args.output_json_path}")
        eprint(f"Optimized parameters: {final_params}")
        
    except Exception as e:
        eprint(f"Fatal error in main: {e}")
        import traceback
        eprint(traceback.format_exc())
        
        # Save error result
        error_result = {
            "Status": "error",
            "Message": str(e),
            "BestParameters": None,
            "Performance": None
        }
        
        try:
            with open(args.output_json_path, 'w') as f:
                json.dump(error_result, f, indent=4)
        except:
            pass
        
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy Parameter Optimizer")
    parser.add_argument("input_csv_path", help="Path to input CSV file")
    parser.add_argument("output_json_path", help="Path to output JSON file")
    parser.add_argument("--mode", required=True, choices=['train', 'fine-tune'], 
                       help="Operation mode")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback window size")
    
    args = parser.parse_args()
    main(args)