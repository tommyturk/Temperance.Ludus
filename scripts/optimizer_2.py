#!/usr/bin/env python3
"""
GPU-Accelerated Optimizer - Fixed version for CuPy compatibility
Uses CuPy for GPU acceleration with workarounds for unsupported operations
"""

import sys
import json
import os
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import itertools
import time

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import GPU libraries
try:
    import cupy as cp
    HAS_CUPY = True
    print("[GPU] CuPy available - using GPU acceleration", file=sys.stderr)
except ImportError:
    HAS_CUPY = False
    print("[WARNING] CuPy not available - falling back to CPU", file=sys.stderr)

def eprint(*args, **kwargs):
    """Print to stderr for logging"""
    print(*args, file=sys.stderr, **kwargs)

def setup_gpu():
    """Configure GPU with memory growth and optimizations"""
    try:
        import tensorflow as tf
        
        # Enable memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable XLA compilation for speedup
            tf.config.optimizer.set_jit(True)
            
            # Mixed precision for 2x speedup on modern GPUs
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            eprint(f"[OK] Configured {len(gpus)} GPU(s) with memory growth and mixed precision")
            
            # If we have CuPy, configure it too
            if HAS_CUPY:
                # Use the first GPU by default
                cp.cuda.Device(0).use()
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                eprint(f"[OK] CuPy using GPU 0 with memory pool")
            
            return True, tf
        else:
            eprint("[WARNING] No GPU detected, using CPU")
            import tensorflow as tf
            return False, tf
    except Exception as e:
        eprint(f"[ERROR] GPU setup failed: {e}")
        import tensorflow as tf
        return False, tf

# Initialize GPU
HAS_GPU, tf = setup_gpu()

# ============================================================================
# PARAMETER GRID CONFIGURATION (same as original)
# ============================================================================

PARAM_GRID = {
    'ma_period': list(range(10, 31, 2)),  # 11 values
    'std_dev_multiplier': [round(x, 1) for x in np.arange(1.5, 3.1, 0.2)],  # 9 values
    'rsi_period': list(range(10, 31, 2)),  # 11 values
    'rsi_oversold': list(range(20, 36, 2)),  # 8 values
    'rsi_overbought': list(range(65, 81, 2)),  # 8 values
    'atr_period': list(range(10, 31, 2)),  # 11 values
    'atr_multiplier': [round(x, 1) for x in np.arange(1.0, 3.1, 0.2)]  # 11 values
}

FEATURE_NAMES = [
    'return_1d', 'return_5d', 'return_10d',
    'volatility_5d', 'volatility_20d', 'volatility_60d',
    'atr_14', 'atr_20',
    'rsi_14', 'rsi_28',
    'bb_width_20',
    'volume_ratio',
    'hl_range_norm',
    'price_position',
    'momentum_10d'
]

# ============================================================================
# GPU-ACCELERATED TECHNICAL INDICATORS
# ============================================================================

def calculate_rsi_gpu(prices, period=14):
    """Calculate RSI indicator on GPU"""
    if HAS_CUPY:
        prices_gpu = cp.asarray(prices)
        delta = cp.diff(prices_gpu, prepend=prices_gpu[0])
        gains = cp.maximum(delta, 0)
        losses = -cp.minimum(delta, 0)
        
        # Calculate EMA on GPU
        alpha = 1.0 / period
        avg_gain = cp.zeros_like(prices_gpu)
        avg_loss = cp.zeros_like(prices_gpu)
        
        avg_gain[0] = cp.mean(gains[:period]) if period < len(gains) else cp.mean(gains)
        avg_loss[0] = cp.mean(losses[:period]) if period < len(losses) else cp.mean(losses)
        
        for i in range(1, len(prices_gpu)):
            avg_gain[i] = alpha * gains[i] + (1 - alpha) * avg_gain[i-1]
            avg_loss[i] = alpha * losses[i] + (1 - alpha) * avg_loss[i-1]
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Transfer back to CPU
        return cp.asnumpy(rsi).astype(np.float32)
    else:
        # CPU fallback (original implementation)
        delta = np.diff(prices, prepend=prices[0])
        gains = np.maximum(delta, 0)
        losses = -np.minimum(delta, 0)
        
        avg_gain = pd.Series(gains).rolling(window=period, min_periods=1).mean()
        avg_loss = pd.Series(losses).rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50).values

def calculate_atr_gpu(high, low, close, period=14):
    """Calculate Average True Range on GPU"""
    if HAS_CUPY:
        high_gpu = cp.asarray(high)
        low_gpu = cp.asarray(low)
        close_gpu = cp.asarray(close)
        
        prev_close = cp.roll(close_gpu, 1)
        prev_close[0] = close_gpu[0]
        
        hl = high_gpu - low_gpu
        hc = cp.abs(high_gpu - prev_close)
        lc = cp.abs(low_gpu - prev_close)
        
        tr = cp.maximum(hl, cp.maximum(hc, lc))
        
        # Moving average on GPU
        kernel = cp.ones(period) / period
        atr = cp.convolve(tr, kernel, mode='same')
        
        return cp.asnumpy(atr).astype(np.float32)
    else:
        # CPU fallback
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        hl = high - low
        hc = np.abs(high - prev_close)
        lc = np.abs(low - prev_close)
        
        tr = np.maximum(hl, np.maximum(hc, lc))
        atr = pd.Series(tr).rolling(window=period, min_periods=1).mean()
        
        return atr.fillna(0).values

# ============================================================================
# UTILITY FUNCTION FOR RUNNING MAXIMUM (CuPy workaround)
# ============================================================================

def running_maximum_gpu(arr):
    """
    Calculate running maximum on GPU
    Workaround for cp.maximum.accumulate not being supported
    """
    if HAS_CUPY and isinstance(arr, cp.ndarray):
        result = cp.empty_like(arr)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = cp.maximum(result[i-1], arr[i])
        return result
    else:
        # NumPy fallback
        return np.maximum.accumulate(arr)

# ============================================================================
# GPU-ACCELERATED BACKTESTING ENGINE
# ============================================================================

def vectorized_backtest_gpu(prices, entries, exits):
    """
    GPU-accelerated vectorized backtesting with CuPy workarounds
    """
    if HAS_CUPY:
        try:
            # Transfer to GPU
            prices_gpu = cp.asarray(prices)
            entries_gpu = cp.asarray(entries)
            exits_gpu = cp.asarray(exits)
            
            n = len(prices_gpu)
            if n < 2:
                return get_empty_metrics()
            
            # Create signals on GPU
            signals = cp.zeros(n, dtype=cp.int8)
            if len(entries_gpu) > 0:
                valid_entries = entries_gpu[entries_gpu < n]
                if len(valid_entries) > 0:
                    signals[valid_entries] = 1
            
            if len(exits_gpu) > 0:
                valid_exits = exits_gpu[exits_gpu < n]
                if len(valid_exits) > 0:
                    signals[valid_exits] = -1
            
            # Build positions on GPU
            positions = cp.zeros(n, dtype=cp.float32)
            current_pos = 0.0
            
            for i in range(n):
                if signals[i] == 1:
                    current_pos = 1.0
                elif signals[i] == -1:
                    current_pos = 0.0
                positions[i] = current_pos
            
            # Calculate returns on GPU
            price_returns = cp.zeros(n)
            price_returns[1:] = cp.diff(prices_gpu) / prices_gpu[:-1]
            
            strategy_returns = positions * price_returns
            
            # Calculate metrics on GPU
            total_return = float(cp.prod(1 + strategy_returns) - 1)
            
            if cp.std(strategy_returns) > 0:
                sharpe_ratio = float(cp.sqrt(252) * cp.mean(strategy_returns) / cp.std(strategy_returns))
            else:
                sharpe_ratio = 0.0
            
            # Number of trades
            position_changes = cp.diff(positions)
            num_trades = int(cp.sum(cp.abs(position_changes) > 0.5) / 2)
            
            # Win rate
            trade_returns = strategy_returns[strategy_returns != 0]
            if len(trade_returns) > 0:
                win_rate = float(cp.sum(trade_returns > 0) / len(trade_returns))
            else:
                win_rate = 0.0
            
            # Max drawdown - using workaround for maximum.accumulate
            cumulative = cp.cumprod(1 + strategy_returns)
            running_max = running_maximum_gpu(cumulative)
            drawdown = (cumulative - running_max) / (running_max + 1e-10)
            max_drawdown = float(cp.min(drawdown))
            
            # Profit factor
            gains = strategy_returns[strategy_returns > 0]
            losses = -strategy_returns[strategy_returns < 0]
            if len(losses) > 0 and cp.sum(losses) > 0:
                profit_factor = float(cp.sum(gains) / cp.sum(losses))
            else:
                profit_factor = 0.0 if len(gains) == 0 else float(cp.sum(gains))
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'profit_factor': profit_factor
            }
        except Exception as e:
            eprint(f"[WARNING] GPU backtest failed: {e}, falling back to CPU")
            return vectorized_backtest_cpu(prices, entries, exits)
    else:
        # CPU fallback
        return vectorized_backtest_cpu(prices, entries, exits)

def vectorized_backtest_cpu(prices, entries, exits):
    """Original CPU implementation for fallback"""
    try:
        prices_arr = np.asarray(prices)
        if len(prices_arr) < 2:
            return get_empty_metrics()
        
        n = len(prices_arr)
        
        # Create signals
        signals = np.zeros(n, dtype=np.int8)
        
        if len(entries) > 0:
            valid_entries = entries[entries < n]
            if len(valid_entries) > 0:
                signals[valid_entries] = 1
        
        if len(exits) > 0:
            valid_exits = exits[exits < n]
            if len(valid_exits) > 0:
                signals[valid_exits] = -1
        
        # Build positions
        positions = np.zeros(n, dtype=np.float32)
        current_pos = 0.0
        
        for i in range(n):
            if signals[i] == 1:
                current_pos = 1.0
            elif signals[i] == -1:
                current_pos = 0.0
            positions[i] = current_pos
        
        # Calculate returns
        price_returns = np.zeros(n)
        price_returns[1:] = np.diff(prices_arr) / prices_arr[:-1]
        
        strategy_returns = positions * price_returns
        
        # Metrics
        total_return = float(np.prod(1 + strategy_returns) - 1)
        
        if np.std(strategy_returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
        else:
            sharpe_ratio = 0
        
        position_changes = np.diff(positions)
        num_trades = int(np.sum(np.abs(position_changes) > 0.5) / 2)
        
        trade_returns = strategy_returns[strategy_returns != 0]
        if len(trade_returns) > 0:
            win_rate = np.sum(trade_returns > 0) / len(trade_returns)
        else:
            win_rate = 0
        
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-10)
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        gains = strategy_returns[strategy_returns > 0]
        losses = -strategy_returns[strategy_returns < 0]
        if len(losses) > 0 and np.sum(losses) > 0:
            profit_factor = np.sum(gains) / np.sum(losses)
        else:
            profit_factor = 0 if len(gains) == 0 else np.sum(gains)
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'num_trades': int(num_trades),
            'win_rate': float(win_rate),
            'max_drawdown': float(max_drawdown),
            'profit_factor': float(profit_factor)
        }
    except Exception as e:
        eprint(f"[ERROR] CPU backtest failed: {e}")
        return get_empty_metrics()

def get_empty_metrics():
    """Return empty metrics dictionary"""
    return {
        'total_return': 0.0,
        'sharpe_ratio': 0.0,
        'num_trades': 0,
        'win_rate': 0.0,
        'max_drawdown': 0.0,
        'profit_factor': 0.0
    }

# ============================================================================
# GPU-ACCELERATED PARAMETER OPTIMIZATION
# ============================================================================

def optimize_parameters_gpu(df, param_grid=None, batch_size=10000):
    """
    GPU-accelerated parameter optimization
    Process parameters in large batches on GPU
    """
    if param_grid is None:
        param_grid = PARAM_GRID
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(*[param_grid[k] for k in param_grid.keys()]))
    total_combinations = len(param_combinations)
    
    eprint(f"\n[GPU] Starting optimization with {total_combinations:,} parameter combinations")
    eprint(f"[GPU] Processing in batches of {batch_size:,}")
    
    # Prepare data
    prices = df['ClosePrice'].values
    high = df['HighPrice'].values
    low = df['LowPrice'].values
    
    best_params = None
    best_sharpe = -float('inf')
    
    start_time = time.time()
    
    # Process in GPU-sized batches
    for batch_start in range(0, total_combinations, batch_size):
        batch_end = min(batch_start + batch_size, total_combinations)
        batch_params = param_combinations[batch_start:batch_end]
        
        # Process batch
        for idx, params_tuple in enumerate(batch_params):
            params = dict(zip(param_grid.keys(), params_tuple))
            
            # Generate signals for this parameter set
            ma_period = params['ma_period']
            std_multiplier = params['std_dev_multiplier']
            rsi_period = params['rsi_period']
            rsi_oversold = params['rsi_oversold']
            rsi_overbought = params['rsi_overbought']
            atr_period = params['atr_period']
            atr_multiplier = params['atr_multiplier']
            
            # Calculate indicators
            ma = pd.Series(prices).rolling(window=ma_period, min_periods=1).mean().values
            std = pd.Series(prices).rolling(window=ma_period, min_periods=1).std().fillna(0).values
            upper_band = ma + (std * std_multiplier)
            lower_band = ma - (std * std_multiplier)
            
            rsi = calculate_rsi_gpu(prices, rsi_period)
            atr = calculate_atr_gpu(high, low, prices, atr_period)
            
            # Generate signals
            entries = np.where((prices < lower_band) & (rsi < rsi_oversold))[0]
            exits = np.where((prices > upper_band) | (rsi > rsi_overbought))[0]
            
            # Backtest (GPU-accelerated)
            metrics = vectorized_backtest_gpu(prices, entries, exits)
            
            # Track best
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_params = params
                best_metrics = metrics
        
        # Progress update every 10% or at end
        if (batch_end % (total_combinations // 10) == 0) or (batch_end == total_combinations):
            elapsed = time.time() - start_time
            processed = batch_end
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total_combinations - processed) / rate if rate > 0 else 0
            
            eprint(f"[GPU] Processed {processed:,}/{total_combinations:,} "
                  f"({100*processed/total_combinations:.1f}%) - "
                  f"Rate: {rate:.0f}/sec - ETA: {remaining:.0f}s")
    
    total_time = time.time() - start_time
    eprint(f"[GPU] Optimization complete in {total_time:.1f}s "
          f"({total_combinations/total_time:.0f} combinations/sec)")
    
    return best_params, best_metrics

# ============================================================================
# FEATURE EXTRACTION (keeping original implementation)
# ============================================================================

def extract_features(df, lookback=60):
    """Extract features for ML model"""
    close = df['ClosePrice'].copy()
    high = df['HighPrice'].copy()
    low = df['LowPrice'].copy()
    volume = df['Volume'].copy()
    
    features = pd.DataFrame(index=df.index)
    
    # Returns (3 features)
    features['return_1d'] = close.pct_change(1)
    features['return_5d'] = close.pct_change(5)
    features['return_10d'] = close.pct_change(10)
    
    # Volatility (3 features)
    features['volatility_5d'] = close.pct_change().rolling(5, min_periods=1).std()
    features['volatility_20d'] = close.pct_change().rolling(20, min_periods=1).std()
    features['volatility_60d'] = close.pct_change().rolling(60, min_periods=1).std()
    
    # ATR (2 features) - using GPU acceleration
    features['atr_14'] = calculate_atr_gpu(high.values, low.values, close.values, 14) / close.values
    features['atr_20'] = calculate_atr_gpu(high.values, low.values, close.values, 20) / close.values
    
    # RSI (2 features) - using GPU acceleration
    features['rsi_14'] = calculate_rsi_gpu(close.values, 14) / 100.0
    features['rsi_28'] = calculate_rsi_gpu(close.values, 28) / 100.0
    
    # Bollinger Band width (1 feature)
    rolling_mean = close.rolling(20, min_periods=1).mean()
    rolling_std = close.rolling(20, min_periods=1).std()
    features['bb_width_20'] = (2 * rolling_std) / (rolling_mean + 1e-10)
    
    # Volume (1 feature)
    features['volume_ratio'] = volume / volume.rolling(20, min_periods=1).mean()
    
    # Price range (1 feature)
    features['hl_range_norm'] = (high - low) / (close + 1e-10)
    
    # Price position in range (1 feature)
    rolling_min = close.rolling(20, min_periods=1).min()
    rolling_max = close.rolling(20, min_periods=1).max()
    features['price_position'] = (close - rolling_min) / (rolling_max - rolling_min + 1e-10)
    
    # Momentum (1 feature)
    features['momentum_10d'] = close / close.shift(10).fillna(close.iloc[0]) - 1
    
    # Fill NaN and clip extreme values
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    features = features.clip(-10, 10)
    
    return features.astype(np.float32)

# ============================================================================
# TRAINING SAMPLE GENERATION (GPU-accelerated)
# ============================================================================

def generate_training_samples(df, window_size=60, stride=10, param_grid=None, max_samples=10000):
    """
    Generate training samples using GPU-accelerated backtesting
    Limit samples to prevent memory issues
    """
    if param_grid is None:
        param_grid = PARAM_GRID
    
    eprint(f"\n[GPU] Generating training samples (max {max_samples:,})...")
    
    # Extract features
    features_df = extract_features(df, lookback=window_size)
    
    X_samples = []
    y_samples = []
    sample_metrics = []
    
    # Sample parameter combinations instead of using all
    all_param_combinations = list(itertools.product(*[param_grid[k] for k in param_grid.keys()]))
    
    if len(all_param_combinations) > max_samples:
        import random
        random.seed(42)  # For reproducibility
        sampled_combinations = random.sample(all_param_combinations, max_samples)
        eprint(f"[GPU] Sampling {len(sampled_combinations):,} from {len(all_param_combinations):,} total combinations")
    else:
        sampled_combinations = all_param_combinations
    
    # Process each window position
    windows_processed = 0
    for start_idx in range(0, len(df) - window_size - 30, stride):
        window_end = start_idx + window_size
        
        # Feature window
        feature_window = features_df.iloc[start_idx:window_end]
        if len(feature_window) < window_size:
            continue
        
        # Validation period (next 30 days)
        val_start = window_end
        val_end = min(window_end + 30, len(df))
        val_df = df.iloc[val_start:val_end]
        
        if len(val_df) < 10:  # Need minimum data for validation
            continue
        
        # Get features for this window
        X = feature_window[FEATURE_NAMES].values
        
        # Test sampled parameters on validation period (limit to 100 per window for speed)
        best_sharpe = -float('inf')
        best_params = None
        
        # Sample a subset for this window
        window_samples = min(100, len(sampled_combinations))
        window_combinations = sampled_combinations[:window_samples]
        
        for params_tuple in window_combinations:
            params = dict(zip(param_grid.keys(), params_tuple))
            
            # Quick backtest on validation period
            prices = val_df['ClosePrice'].values
            high = val_df['HighPrice'].values
            low = val_df['LowPrice'].values
            
            # Calculate indicators
            ma = pd.Series(prices).rolling(window=params['ma_period'], min_periods=1).mean().values
            std = pd.Series(prices).rolling(window=params['ma_period'], min_periods=1).std().fillna(0).values
            upper_band = ma + (std * params['std_dev_multiplier'])
            lower_band = ma - (std * params['std_dev_multiplier'])
            
            rsi = calculate_rsi_gpu(prices, params['rsi_period'])
            
            # Generate signals
            entries = np.where((prices < lower_band) & (rsi < params['rsi_oversold']))[0]
            exits = np.where((prices > upper_band) | (rsi > params['rsi_overbought']))[0]
            
            # Backtest
            metrics = vectorized_backtest_gpu(prices, entries, exits)
            
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_params = params
        
        # Store sample
        if best_params is not None and best_sharpe > -float('inf'):
            X_samples.append(X)
            
            # Normalize parameters to 0-1 range for neural network
            y = np.array([
                (best_params['ma_period'] - 10) / 20,
                (best_params['std_dev_multiplier'] - 1.5) / 1.5,
                (best_params['rsi_period'] - 10) / 20,
                (best_params['rsi_oversold'] - 20) / 15,
                (best_params['rsi_overbought'] - 65) / 15,
                (best_params['atr_period'] - 10) / 20,
                (best_params['atr_multiplier'] - 1.0) / 2.0
            ], dtype=np.float32)
            
            y_samples.append(y)
            sample_metrics.append({'validation_sharpe': best_sharpe})
            
        windows_processed += 1
        if windows_processed % 10 == 0:
            eprint(f"[GPU] Processed {windows_processed} windows, generated {len(X_samples)} samples")
    
    eprint(f"[GPU] Generated {len(X_samples)} training samples from {windows_processed} windows")
    
    if len(X_samples) == 0:
        return np.array([]), np.array([]), []
    
    return np.array(X_samples), np.array(y_samples), sample_metrics

# ============================================================================
# MODEL TRAINING (keeping TensorFlow implementation)
# ============================================================================

def create_model(input_shape, output_shape):
    """Create neural network model"""
    from tensorflow.keras import layers, models, regularizers
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # 2. Flatten layer 
        layers.Flatten(),
        # Batch normalization for input
        layers.BatchNormalization(),
        
        # Dense layers with dropout
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        
        # Output layer (7 parameters, normalized to 0-1)
        layers.Dense(output_shape, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32, model_path=None):
    """Train the neural network model"""
    eprint(f"\n[TRAIN] Starting model training with {len(X_train)} samples")
    
    # Create model
    model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_shape=y_train.shape[1])
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    if model_path:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ))
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def fine_tune_model(model_path, X_tune, y_tune, epochs=20, output_path=None):
    """Fine-tune existing model"""
    eprint(f"\n[FINE-TUNE] Loading model from {model_path}")
    
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    
    # Use lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    
    # Split for validation
    split_idx = int(0.8 * len(X_tune))
    X_train = X_tune[:split_idx]
    y_train = y_tune[:split_idx]
    X_val = X_tune[split_idx:]
    y_val = y_tune[split_idx:]
    
    # Fine-tune
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=epochs,
        batch_size=16,
        verbose=1
    )
    
    # Save
    if output_path:
        model.save(output_path)
        eprint(f"[FINE-TUNE] Model saved to {output_path}")
    
    return model, history

def predict_parameters(model, features_window, param_grid):
    """Predict optimal parameters using trained model"""
    # Reshape for model input
    X = features_window.reshape(1, features_window.shape[0], features_window.shape[1])
    
    # Predict normalized parameters
    y_pred = model.predict(X, verbose=0)[0]
    
    # Denormalize parameters
    predicted_params = {
        'MovingAveragePeriod': int(y_pred[0] * 20 + 10),
        'StdDevMultiplier': float(y_pred[1] * 1.5 + 1.5),
        'RSIPeriod': int(y_pred[2] * 20 + 10),
        'RSIOversold': int(y_pred[3] * 15 + 20),
        'RSIOverbought': int(y_pred[4] * 15 + 65),
        'AtrPeriod': int(y_pred[5] * 20 + 10),
        'AtrMultiplier': float(y_pred[6] * 2.0 + 1.0)
    }
    
    return predicted_params

def calculate_baseline_params(param_grid):
    """Calculate baseline (median) parameters"""
    return {
        'MovingAveragePeriod': 20,
        'StdDevMultiplier': 2.0,
        'RSIPeriod': 14,
        'RSIOversold': 30,
        'RSIOverbought': 70,
        'AtrPeriod': 14,
        'AtrMultiplier': 1.5
    }

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(args):
    """Main execution function"""
    try:
        eprint(f"\n{'='*60}")
        eprint(f"Deep Learning Parameter Optimizer (GPU-Accelerated)")
        eprint(f"Mode: {args.mode}")
        eprint(f"Input: {args.input_csv_path}")
        eprint(f"Output: {args.output_json_path}")
        eprint(f"Model: {args.model_path}")
        eprint(f"GPU Available: {HAS_GPU}")
        eprint(f"CuPy Available: {HAS_CUPY}")
        eprint(f"{'='*60}\n")
        
        # Load data
        eprint("[DATA] Loading price data...")
        df = pd.read_csv(args.input_csv_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp').reset_index(drop=True)
        eprint(f"[DATA] Loaded {len(df)} price records from {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        
        # Check data sufficiency
        if len(df) < 150:
            raise ValueError(f"Insufficient data: {len(df)} rows (minimum 150 required)")
        
        result = {}
        
        if args.mode == 'train':
            eprint("\n--- TRAIN MODE ---")
            
            # Split data
            split_idx = int(0.7 * len(df))
            train_df = df.iloc[:split_idx]
            val_df = df.iloc[split_idx:]
            
            eprint(f"[SPLIT] Training: {len(train_df)} rows, Validation: {len(val_df)} rows")
            
            # Generate training samples with GPU acceleration
            X_train, y_train, train_metrics = generate_training_samples(
                train_df,
                window_size=args.lookback,
                stride=10,
                param_grid=PARAM_GRID,
                max_samples=10000  # Limit for faster training
            )
            
            if len(X_train) < 10:
                raise ValueError(f"Insufficient training samples: {len(X_train)}")
            
            # Generate validation samples
            X_val, y_val, val_metrics = generate_training_samples(
                val_df,
                window_size=args.lookback,
                stride=15,
                param_grid=PARAM_GRID,
                max_samples=2000
            )
            
            eprint(f"[SAMPLES] Training: {len(X_train)}, Validation: {len(X_val)}")
            
            # Train model
            model, history = train_model(
                X_train, y_train,
                X_val, y_val,
                epochs=args.epochs,
                batch_size=32,
                model_path=args.model_path
            )
            
            # Prepare result
            avg_sharpe = np.mean([m['validation_sharpe'] for m in train_metrics]) if train_metrics else 0.0
            avg_val_sharpe = np.mean([m['validation_sharpe'] for m in val_metrics]) if val_metrics else 0.0
            
            # Make prediction for latest data
            features = extract_features(df)
            features_latest = features.iloc[-args.lookback:].values
            predicted_params_normalized = predict_parameters(model, features_latest, PARAM_GRID)
           
            # --- NEW: Run backtest with predicted parameters on validation data ---
            eprint(f"[METRICS] Running final backtest on validation data with predicted params: {predicted_params_normalized}")
            val_prices = val_df['ClosePrice'].values
            val_high = val_df['HighPrice'].values
            val_low = val_df['LowPrice'].values

            # Recalculate indicators for validation period using predicted params
            ma_val = pd.Series(val_prices).rolling(window=predicted_params_normalized['MovingAveragePeriod'], min_periods=1).mean().values
            std_val = pd.Series(val_prices).rolling(window=predicted_params_normalized['MovingAveragePeriod'], min_periods=1).std().fillna(0).values
            upper_band_val = ma_val + (std_val * predicted_params_normalized['StdDevMultiplier'])
            lower_band_val = ma_val - (std_val * predicted_params_normalized['StdDevMultiplier'])
            rsi_val = calculate_rsi_gpu(val_prices, predicted_params_normalized['RSIPeriod'])
            # atr_val = calculate_atr_gpu(val_high, val_low, val_prices, predicted_params_normalized['AtrPeriod']) # ATR not used in signals here, omit if not needed

            # Generate signals on validation data
            entries_val = np.where((val_prices < lower_band_val) & (rsi_val < predicted_params_normalized['RSIOversold']))[0]
            exits_val = np.where((val_prices > upper_band_val) | (rsi_val > predicted_params_normalized['RSIOverbought']))[0]

            # Calculate final metrics using GPU backtest
            final_metrics = vectorized_backtest_gpu(val_prices, entries_val, exits_val)
            eprint(f"[METRICS] Final backtest metrics on validation data: {final_metrics}")
            # --- END NEW ---

            result = {
                "Status": "success",
                "BestParameters": predicted_params_normalized,
                "Metrics": {
                    "TotalReturns": final_metrics['total_return'],  # Use actual metrics
                    "SharpeRatio": final_metrics['sharpe_ratio'],   # Use actual metrics
                    "WinRate": final_metrics['win_rate'],           # Use actual metrics
                    "NumTrades": final_metrics['num_trades'],       # Use actual metrics
                    "MaxDrawdown": final_metrics['max_drawdown'],   # Use actual metrics
                    "ProfitFactor": final_metrics['profit_factor']
                },
                "TrainingInfo": {
                    "NumSamples": len(X_train),
                    "AvgTrainSharpe": float(avg_sharpe),
                    "AvgValSharpe": float(avg_val_sharpe),
                    "FinalLoss": float(min(history.history['val_loss'])) if 'val_loss' in history.history else 0.0
                },
                "Message": f"Training complete with {len(X_train)} samples (GPU-accelerated)"
            }
        
        elif args.mode == 'fine-tune':
            eprint("\n--- FINE-TUNE MODE ---")
            
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Model not found: {args.model_path}")
            
            # Generate fine-tuning samples
            X_tune, y_tune, metrics = generate_training_samples(
                df,
                window_size=args.lookback,
                stride=5,
                param_grid=PARAM_GRID,
                max_samples=5000
            )
            
            if len(X_tune) < 5:
                eprint("[WARNING] Insufficient samples for fine-tuning, using existing model")
                from tensorflow.keras.models import load_model
                model = load_model(args.model_path)
            else:
                # Fine-tune
                model, history = fine_tune_model(
                    args.model_path,
                    X_tune, y_tune,
                    epochs=min(args.epochs, 20),
                    output_path=args.model_path
                )
            
            # Make prediction
            features = extract_features(df)
            features_latest = features.iloc[-args.lookback:].values
            predicted_params_normalized = predict_parameters(model, features_latest, PARAM_GRID) # Get denormalized params            
            
            avg_val_sharpe = np.mean([m['validation_sharpe'] for m in metrics]) if metrics else 0.0
            
            # --- NEW: Run backtest with predicted parameters on latest data (e.g., last 200 periods) ---
            eprint(f"[METRICS] Running final backtest on latest data with predicted params: {predicted_params_normalized}")
            latest_df = df # Or use full df if appropriate
            latest_prices = latest_df['ClosePrice'].values
            latest_high = latest_df['HighPrice'].values
            latest_low = latest_df['LowPrice'].values

            # Recalculate indicators for latest period using predicted params
            ma_latest = pd.Series(latest_prices).rolling(window=predicted_params_normalized['MovingAveragePeriod'], min_periods=1).mean().values
            std_latest = pd.Series(latest_prices).rolling(window=predicted_params_normalized['MovingAveragePeriod'], min_periods=1).std().fillna(0).values
            upper_band_latest = ma_latest + (std_latest * predicted_params_normalized['StdDevMultiplier'])
            lower_band_latest = ma_latest - (std_latest * predicted_params_normalized['StdDevMultiplier'])
            rsi_latest = calculate_rsi_gpu(latest_prices, predicted_params_normalized['RSIPeriod'])

            # Generate signals on latest data
            entries_latest = np.where((latest_prices < lower_band_latest) & (rsi_latest < predicted_params_normalized['RSIOversold']))[0]
            exits_latest = np.where((latest_prices > upper_band_latest) | (rsi_latest > predicted_params_normalized['RSIOverbought']))[0]

            # Calculate final metrics using GPU backtest
            final_metrics = vectorized_backtest_gpu(latest_prices, entries_latest, exits_latest)
            eprint(f"[METRICS] Final backtest metrics on latest data: {final_metrics}")
             # --- END NEW ---

            result = {
                "Status": "success",
                "BestParameters": predicted_params_normalized,
                "Metrics": {
                    "TotalReturns": final_metrics['total_return'],  # Use actual metrics
                    "SharpeRatio": final_metrics['sharpe_ratio'],   # Use actual metrics
                    "WinRate": final_metrics['win_rate'],           # Use actual metrics
                    "NumTrades": final_metrics['num_trades'],       # Use actual metrics
                    "MaxDrawdown": final_metrics['max_drawdown'],   # Use actual metrics
                    "ProfitFactor": final_metrics['profit_factor']  # Use actual metrics
                },
                "FineTuneInfo": {
                    "NumSamples": len(X_tune),
                    "AvgValSharpe": float(avg_val_sharpe)
                },
                "Message": f"Fine-tuning complete with {len(X_tune)} samples (GPU-accelerated)"
            }
        
        else:  # predict mode
            eprint("\n--- PREDICTION MODE ---")
            
            if not os.path.exists(args.model_path):
                # Use GPU-accelerated optimization if no model exists
                eprint("[WARNING] Model not found, using GPU-accelerated optimization")
                
                best_params, best_metrics = optimize_parameters_gpu(
                    df.iloc[-200:] if len(df) > 200 else df,
                    param_grid=PARAM_GRID,
                    batch_size=10000
                )
                
                if best_params is None:
                    raise ValueError("Optimization failed to find valid parameters")
                
                predicted_params = {
                    'MovingAveragePeriod': best_params['ma_period'],
                    'StdDevMultiplier': best_params['std_dev_multiplier'],
                    'RSIPeriod': best_params['rsi_period'],
                    'RSIOversold': best_params['rsi_oversold'],
                    'RSIOverbought': best_params['rsi_overbought'],
                    'AtrPeriod': best_params['atr_period'],
                    'AtrMultiplier': best_params['atr_multiplier']
                }
                
                result = {
                     "Status": "success",
                     "BestParameters": predicted_params,
                     "Metrics": {
                        "TotalReturns": final_metrics['total_return'],  # Use actual metrics
                        "SharpeRatio": final_metrics['sharpe_ratio'],   # Use actual metrics
                        "WinRate": final_metrics['win_rate'],           # Use actual metrics
                        "NumTrades": final_metrics['num_trades'],       # Use actual metrics
                        "MaxDrawdown": final_metrics['max_drawdown'],   # Use actual metrics
                        "ProfitFactor": final_metrics['profit_factor']  # Use actual metrics
                    },
                     "Message": "GPU-accelerated optimization complete as model not found"
                 }
            else:
                # Load model and predict
                from tensorflow.keras.models import load_model
                model = load_model(args.model_path)
                eprint(f"[OK] Loaded model from: {args.model_path}")
                
                # Extract features
                features = extract_features(df)
                if len(features) < args.lookback:
                    raise ValueError(f"Insufficient data for prediction")
                
                # Predict parameters
                features_latest = features.iloc[-args.lookback:].values
                predicted_params_normalized = predict_parameters(model, features_latest, PARAM_GRID)
                eprint(f"[OK] Predicted parameters: {predicted_params_normalized}")
                
                eprint(f"[METRICS] Running final backtest on latest data with predicted params: {predicted_params_normalized}")
                latest_df = df.iloc[-200:] # Use recent data for metrics context
                latest_prices = latest_df['ClosePrice'].values
                latest_high = latest_df['HighPrice'].values
                latest_low = latest_df['LowPrice'].values

                # Recalculate indicators
                ma_latest = pd.Series(latest_prices).rolling(window=predicted_params_normalized['MovingAveragePeriod'], min_periods=1).mean().values
                std_latest = pd.Series(latest_prices).rolling(window=predicted_params_normalized['MovingAveragePeriod'], min_periods=1).std().fillna(0).values
                upper_band_latest = ma_latest + (std_latest * predicted_params_normalized['StdDevMultiplier'])
                lower_band_latest = ma_latest - (std_latest * predicted_params_normalized['StdDevMultiplier'])
                rsi_latest = calculate_rsi_gpu(latest_prices, predicted_params_normalized['RSIPeriod'])

                # Generate signals
                entries_latest = np.where((latest_prices < lower_band_latest) & (rsi_latest < predicted_params_normalized['RSIOversold']))[0]
                exits_latest = np.where((latest_prices > upper_band_latest) | (rsi_latest > predicted_params_normalized['RSIOverbought']))[0]

                # Calculate final metrics
                final_metrics = vectorized_backtest_gpu(latest_prices, entries_latest, exits_latest)
                eprint(f"[METRICS] Final backtest metrics on latest data: {final_metrics}")
                # --- END NEW ---

                result = {
                    "Status": "success",
                    "BestParameters": predicted_params_normalized,
                    "Metrics": {
                        "TotalReturns": final_metrics['total_return'],  # Use actual metrics
                        "SharpeRatio": final_metrics['sharpe_ratio'],   # Use actual metrics
                        "WinRate": final_metrics['win_rate'],           # Use actual metrics
                        "NumTrades": final_metrics['num_trades'],       # Use actual metrics
                        "MaxDrawdown": final_metrics['max_drawdown'],   # Use actual metrics
                        "ProfitFactor": final_metrics['profit_factor']  # Use actual metrics
                    },
                    "Message": "Prediction complete (GPU-accelerated)"
                }
        
        # Save result
        with open(args.output_json_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        eprint(f"\n[OK] Results saved to: {args.output_json_path}")
        eprint(f"[OK] Process complete!\n")
        
    except Exception as e:
        eprint(f"\n[ERROR] Fatal error: {e}")
        import traceback
        eprint(traceback.format_exc())
        
        # Save error result
        error_result = {
            "Status": "error",
            "Message": str(e),
            "BestParameters": None,
            "Metrics": None
        }
        
        try:
            with open(args.output_json_path, 'w') as f:
                json.dump(error_result, f, indent=4)
        except:
            pass
        
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-Accelerated Deep Learning Parameter Optimizer")
    parser.add_argument("input_csv_path", help="Path to input CSV file")
    parser.add_argument("output_json_path", help="Path to output JSON file")
    parser.add_argument("--mode", required=True, choices=['train', 'fine-tune', 'predict'],
                       help="Operation mode")
    parser.add_argument("--model-path", required=True, help="Path to model file (.keras)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback window size")
    
    args = parser.parse_args()
    main(args)