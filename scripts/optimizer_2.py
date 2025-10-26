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

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
# PARAMETER GRID CONFIGURATION
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
# TECHNICAL INDICATORS
# ============================================================================

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50)

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
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

def extract_features(df, lookback=60):
    """
    Extract 15 price-based features for regime detection
    
    Returns: DataFrame with shape (n_samples, 15)
    """
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
    features['volatility_5d'] = close.pct_change().rolling(5).std()
    features['volatility_20d'] = close.pct_change().rolling(20).std()
    features['volatility_60d'] = close.pct_change().rolling(60).std()
    
    # ATR (2 features)
    features['atr_14'] = calculate_atr(df, 14) / close
    features['atr_20'] = calculate_atr(df, 20) / close
    
    # RSI (2 features)
    features['rsi_14'] = calculate_rsi(close, 14) / 100.0
    features['rsi_28'] = calculate_rsi(close, 28) / 100.0
    
    # Bollinger Band width (1 feature)
    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    features['bb_width_20'] = (2 * rolling_std) / rolling_mean
    
    # Volume (1 feature)
    features['volume_ratio'] = volume / volume.rolling(20).mean()
    
    # Price range (1 feature)
    features['hl_range_norm'] = (high - low) / close
    
    # Price position in range (1 feature)
    rolling_min = close.rolling(20).min()
    rolling_max = close.rolling(20).max()
    features['price_position'] = (close - rolling_min) / (rolling_max - rolling_min + 1e-8)
    
    # Momentum (1 feature)
    features['momentum_10d'] = close / close.shift(10) - 1
    
    # Fill NaN and clip extreme values
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    features = features.clip(-10, 10)  # Clip extreme outliers
    
    return features.astype(np.float32)

# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

def vectorized_backtest(prices, entries, exits):
    """
    Fast vectorized backtesting
    
    Returns: dict with comprehensive metrics
    """
    try:
        prices_arr = np.asarray(prices)
        if len(prices_arr) < 2:
            return get_empty_metrics()
        
        # Create signals
        signals = np.zeros(len(prices_arr), dtype=np.int8)
        
        valid_entries = entries[entries < len(prices_arr)]
        valid_exits = exits[exits < len(prices_arr)]
        
        if len(valid_entries) > 0:
            signals[valid_entries] = 1
        if len(valid_exits) > 0:
            signals[valid_exits] = -1
        
        # Build positions
        positions = np.zeros(len(prices_arr), dtype=np.float32)
        current_pos = 0.0
        
        for i in range(len(signals)):
            if signals[i] == 1:
                current_pos = 1.0
            elif signals[i] == -1:
                current_pos = 0.0
            positions[i] = current_pos
        
        # Shift to avoid look-ahead bias
        positions_shifted = np.roll(positions, 1)
        positions_shifted[0] = 0.0
        
        # Calculate returns
        price_ratios = prices_arr[1:] / prices_arr[:-1]
        price_ratios = np.where(price_ratios <= 0, 1.0, price_ratios)
        log_returns = np.log(price_ratios)
        log_returns = np.concatenate([np.array([0.0]), log_returns])
        
        strategy_returns = positions_shifted * log_returns
        
        # Total return
        total_log_return = np.sum(strategy_returns)
        total_return = np.exp(total_log_return) - 1.0 if np.isfinite(total_log_return) else 0.0
        
        # Sharpe ratio (annualized for hourly data: sqrt(252*6.5))
        non_zero_returns = strategy_returns[strategy_returns != 0]
        if len(non_zero_returns) > 1:
            ret_std = np.std(non_zero_returns)
            ret_mean = np.mean(non_zero_returns)
            sharpe_ratio = (ret_mean / ret_std) * np.sqrt(1638) if ret_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Trade-level metrics
        trades = []
        in_position = False
        entry_price = 0.0
        
        for i in range(len(signals)):
            if signals[i] == 1 and not in_position:
                entry_price = prices_arr[i]
                in_position = True
            elif signals[i] == -1 and in_position:
                exit_price = prices_arr[i]
                trade_return = (exit_price - entry_price) / entry_price
                trades.append(trade_return)
                in_position = False
        
        # Win rate and profit factor
        if len(trades) > 0:
            winning_trades = sum(1 for t in trades if t > 0)
            win_rate = winning_trades / len(trades)
            
            gross_profit = sum(t for t in trades if t > 0)
            gross_loss = abs(sum(t for t in trades if t < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        else:
            win_rate = 0.0
            profit_factor = 0.0
        
        # Max drawdown
        cumulative = np.exp(np.cumsum(strategy_returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'win_rate': float(win_rate),
            'num_trades': len(trades),
            'max_drawdown': float(max_drawdown),
            'profit_factor': float(profit_factor)
        }
        
    except Exception as e:
        eprint(f"[ERROR] Backtest error: {e}")
        return get_empty_metrics()

def get_empty_metrics():
    """Return empty metrics dict"""
    return {
        'total_return': 0.0,
        'sharpe_ratio': 0.0,
        'win_rate': 0.0,
        'num_trades': 0,
        'max_drawdown': 0.0,
        'profit_factor': 0.0
    }

# ============================================================================
# GRID SEARCH OPTIMIZATION
# ============================================================================

def optimize_parameters(df, param_grid):
    """
    Run grid search to find optimal parameters
    
    Returns: (optimal_params_dict, metrics_dict)
    """
    close_prices = df['ClosePrice'].values
    
    param_combinations = list(itertools.product(*param_grid.values()))
    param_keys = list(param_grid.keys())
    
    best_sharpe = -float('inf')
    best_params = None
    best_metrics = None
    
    eprint(f"  Testing {len(param_combinations)} parameter combinations...")
    
    for params in param_combinations:
        p = dict(zip(param_keys, params))
        
        try:
            # Calculate indicators
            mean = pd.Series(close_prices).rolling(window=p['ma_period'], min_periods=p['ma_period']).mean()
            std = pd.Series(close_prices).rolling(window=p['ma_period'], min_periods=p['ma_period']).std()
            rsi = calculate_rsi(pd.Series(close_prices), p['rsi_period'])
            atr = calculate_atr(df, p['atr_period'])
            
            valid_start = max(p['ma_period'], p['rsi_period'], p['atr_period'])
            
            lower_band = mean - std * p['std_dev_multiplier'] - atr * p['atr_multiplier']
            
            # Entry/exit conditions
            price_below_lower = close_prices < lower_band
            rsi_oversold = rsi < p['rsi_oversold']
            price_above_mean = close_prices > mean
            
            entry_condition = price_below_lower & rsi_oversold
            exit_condition = price_above_mean
            
            entry_indices = np.where(entry_condition.fillna(False))[0]
            exit_indices = np.where(exit_condition.fillna(False))[0]
            
            entry_indices = entry_indices[entry_indices >= valid_start]
            exit_indices = exit_indices[exit_indices >= valid_start]
            
            if len(entry_indices) >= 2 and len(exit_indices) >= 2:
                metrics = vectorized_backtest(close_prices, entry_indices, exit_indices)
                
                if metrics['sharpe_ratio'] > best_sharpe:
                    best_sharpe = metrics['sharpe_ratio']
                    best_params = p
                    best_metrics = metrics
        
        except Exception:
            continue
    
    if best_params is None:
        eprint("  [WARNING] No valid parameter combination found, using defaults")
        best_params = {k: np.median(v) for k, v in param_grid.items()}
        best_metrics = get_empty_metrics()
    
    return best_params, best_metrics

# ============================================================================
# TRAINING DATA GENERATION
# ============================================================================

def generate_training_samples(df, window_size=60, stride=5, param_grid=PARAM_GRID):
    """
    Generate training samples with rolling windows
    
    Args:
        df: Price DataFrame
        window_size: Size of each window (60 days)
        stride: Days between windows (5 = ~146 samples per 2 years)
        param_grid: Parameter grid for optimization
    
    Returns: (X, y, metrics_list)
    """
    X_samples = []
    y_samples = []
    metrics_list = []
    
    num_windows = (len(df) - window_size - 30) // stride
    eprint(f"Generating {num_windows} training samples (stride={stride})...")
    
    for i in range(0, len(df) - window_size - 30, stride):
        if i % (stride * 10) == 0:
            progress = (i / stride) / num_windows * 100
            eprint(f"  Progress: {progress:.1f}% ({i // stride}/{num_windows})")
        
        # Extract window
        window_df = df.iloc[i:i+window_size].copy()
        validation_df = df.iloc[i+window_size:i+window_size+30].copy()
        
        if len(window_df) < window_size:
            continue
        
        # Extract features
        features = extract_features(window_df)
        if len(features) < window_size or features.isnull().any().any():
            continue
        
        feature_window = features.iloc[-window_size:].values
        
        # Optimize parameters on this window
        optimal_params, train_metrics = optimize_parameters(window_df, param_grid)
        
        # Validate on next 30 days
        val_metrics = get_empty_metrics()
        if len(validation_df) >= 30:
            _, val_metrics = optimize_parameters(validation_df, {k: [v] for k, v in optimal_params.items()})
        
        # Store sample
        param_values = [
            optimal_params['ma_period'],
            optimal_params['std_dev_multiplier'],
            optimal_params['rsi_period'],
            optimal_params['rsi_oversold'],
            optimal_params['rsi_overbought'],
            optimal_params['atr_period'],
            optimal_params['atr_multiplier']
        ]
        
        X_samples.append(feature_window)
        y_samples.append(param_values)
        metrics_list.append({**train_metrics, 'validation_sharpe': val_metrics['sharpe_ratio']})
    
    eprint(f"[OK] Generated {len(X_samples)} valid training samples")
    
    return np.array(X_samples), np.array(y_samples), metrics_list

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def create_model(input_shape=(60, 15), output_dim=7):
    """
    Hybrid CNN-LSTM model with attention
    
    Architecture:
    - CNN branch: Extracts spatial patterns from price features
    - LSTM branch: Captures temporal dependencies
    - Attention: Focuses on relevant historical periods
    - Dense layers: Maps to 7 parameters
    """
    from tensorflow.keras.layers import (
        Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization,
        MaxPooling1D, GlobalMaxPooling1D, concatenate, multiply,
        RepeatVector, Permute, Flatten, Activation, Lambda
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend as K
    
    inputs = Input(shape=input_shape, dtype=tf.float32)
    
    # CNN Branch
    cnn = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    cnn = BatchNormalization()(cnn)
    cnn = Conv1D(128, 3, activation='relu', padding='same')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling1D(2)(cnn)
    cnn = Conv1D(256, 3, activation='relu')(cnn)
    cnn = GlobalMaxPooling1D()(cnn)
    
    # LSTM Branch with Attention
    lstm = LSTM(128, return_sequences=True, dropout=0.3)(inputs)
    lstm = LSTM(128, return_sequences=True, dropout=0.3)(lstm)
    
    # Attention mechanism
    attention = Dense(1, activation='tanh')(lstm)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(128)(attention)
    attention = Permute([2, 1])(attention)
    
    lstm_attended = multiply([lstm, attention])
    lstm_attended = Lambda(lambda x: K.sum(x, axis=1))(lstm_attended)
    
    # Merge branches
    merged = concatenate([cnn, lstm_attended])
    
    # Dense layers
    dense = Dense(256, activation='relu')(merged)
    dense = Dropout(0.4)(dense)
    dense = Dense(128, activation='relu')(dense)
    dense = Dropout(0.3)(dense)
    
    # Output layer (float32 for numerical stability)
    outputs = Dense(output_dim, activation='linear', dtype=tf.float32)(dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# ============================================================================
# TRAINING & FINE-TUNING
# ============================================================================

def train_model(X_train, y_train, epochs=100, model_path=None):
    """Train the model from scratch"""
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    
    eprint(f"\nTraining model on {len(X_train)} samples for {epochs} epochs...")
    
    # Create model
    model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Compile
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1)
    ]
    
    if model_path:
        callbacks.append(ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', verbose=1))
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=min(256, max(32, len(X_train) // 4)),
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    eprint(f"[OK] Training complete. Best val_loss: {min(history.history['val_loss']):.6f}")
    
    return model, history

def fine_tune_model(model_path, X_tune, y_tune, epochs=20, output_path=None):
    """Fine-tune existing model"""
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    eprint(f"\nFine-tuning model on {len(X_tune)} samples for {epochs} epochs...")
    
    # Load existing model
    model = load_model(model_path)
    
    # Freeze first 2 layers
    for layer in model.layers[:2]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    optimizer = Adam(learning_rate=0.0001)  # 10x lower
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]
    
    if output_path:
        callbacks.append(ModelCheckpoint(filepath=output_path, save_best_only=True, monitor='val_loss', verbose=1))
    
    # Fine-tune
    history = model.fit(
        X_tune, y_tune,
        epochs=epochs,
        batch_size=min(128, max(16, len(X_tune) // 4)),
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    eprint(f"[OK] Fine-tuning complete. Best val_loss: {min(history.history['val_loss']):.6f}")
    
    return model, history

# ============================================================================
# PREDICTION & VALIDATION
# ============================================================================

def predict_parameters(model, features_60d, param_grid=PARAM_GRID):
    """
    Predict optimal parameters for given features
    
    Returns: dict with parameter names and values
    """
    # Reshape for prediction
    X_pred = features_60d.reshape(1, features_60d.shape[0], features_60d.shape[1])
    
    # Predict
    predicted = model.predict(X_pred, verbose=0)[0]
    
    # Map to parameter ranges
    param_keys = list(param_grid.keys())
    final_params = {}
    
    for i, key in enumerate(param_keys):
        param_range = param_grid[key]
        raw_value = predicted[i]
        
        if key in ['ma_period', 'rsi_period', 'rsi_oversold', 'rsi_overbought', 'atr_period']:
            clipped_value = int(np.round(np.clip(raw_value, min(param_range), max(param_range))))
        else:
            clipped_value = float(np.round(np.clip(raw_value, min(param_range), max(param_range)), 2))
        
        # Map to output names
        name_mapping = {
            'ma_period': 'MovingAveragePeriod',
            'std_dev_multiplier': 'StdDevMultiplier',
            'rsi_period': 'RSIPeriod',
            'rsi_oversold': 'RSIOversold',
            'rsi_overbought': 'RSIOverbought',
            'atr_period': 'AtrPeriod',
            'atr_multiplier': 'AtrMultiplier'
        }
        
        final_params[name_mapping[key]] = clipped_value
    
    return final_params

def calculate_baseline_params(param_grid):
    """Calculate baseline (median) parameters"""
    baseline = {}
    
    name_mapping = {
        'ma_period': 'MovingAveragePeriod',
        'std_dev_multiplier': 'StdDevMultiplier',
        'rsi_period': 'RSIPeriod',
        'rsi_oversold': 'RSIOversold',
        'rsi_overbought': 'RSIOverbought',
        'atr_period': 'AtrPeriod',
        'atr_multiplier': 'AtrMultiplier'
    }
    
    for key, values in param_grid.items():
        median_val = np.median(values)
        if key in ['ma_period', 'rsi_period', 'rsi_oversold', 'rsi_overbought', 'atr_period']:
            median_val = int(median_val)
        else:
            median_val = round(median_val, 2)
        baseline[name_mapping[key]] = median_val
    
    return baseline

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(args):
    """Main execution function"""
    try:
        eprint(f"\n{'='*60}")
        eprint(f"Deep Learning Regime Optimizer")
        eprint(f"Mode: {args.mode}")
        eprint(f"{'='*60}\n")
        
        # Load data
        eprint(f"Loading data from: {args.input_csv_path}")
        df = pd.read_csv(args.input_csv_path, parse_dates=['Timestamp'])
        df = df.sort_values('Timestamp').reset_index(drop=True)
        eprint(f"[OK] Loaded {len(df)} price records")
        
        if len(df) < 150:
            raise ValueError(f"Insufficient data: need at least 150 rows, got {len(df)}")
        
        # Execute based on mode
        if args.mode == 'train':
            eprint("\n--- TRAINING MODE ---")
            
            # Generate training samples
            X_train, y_train, metrics = generate_training_samples(
                df, 
                window_size=args.lookback,
                stride=5,
                param_grid=PARAM_GRID
            )
            
            if len(X_train) < 10:
                raise ValueError(f"Insufficient training samples generated: {len(X_train)}")
            
            # Train model
            model, history = train_model(X_train, y_train, epochs=args.epochs, model_path=args.model_path)
            
            # Make prediction for validation
            features = extract_features(df)
            features_60d = features.iloc[-args.lookback:].values
            predicted_params = predict_parameters(model, features_60d, PARAM_GRID)
            
            # Calculate metrics
            avg_sharpe = np.mean([m['sharpe_ratio'] for m in metrics])
            avg_val_sharpe = np.mean([m['validation_sharpe'] for m in metrics])
            
            result = {
                "Status": "success",
                "BestParameters": predicted_params,
                "Metrics": {
                    "TotalReturns": None,  # Will be calculated by backtest
                    "SharpeRatio": float(avg_val_sharpe),
                    "WinRate": None,
                    "NumTrades": None,
                    "MaxDrawdown": None,
                    "ProfitFactor": None
                },
                "TrainingInfo": {
                    "NumSamples": len(X_train),
                    "AvgTrainSharpe": float(avg_sharpe),
                    "AvgValSharpe": float(avg_val_sharpe),
                    "FinalLoss": float(min(history.history['val_loss']))
                },
                "Message": f"Training complete with {len(X_train)} samples"
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
                param_grid=PARAM_GRID
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
            features_60d = features.iloc[-args.lookback:].values
            predicted_params = predict_parameters(model, features_60d, PARAM_GRID)
            
            avg_val_sharpe = np.mean([m['validation_sharpe'] for m in metrics]) if metrics else 0.0
            
            result = {
                "Status": "success",
                "BestParameters": predicted_params,
                "Metrics": {
                    "TotalReturns": None,
                    "SharpeRatio": float(avg_val_sharpe),
                    "WinRate": None,
                    "NumTrades": None,
                    "MaxDrawdown": None,
                    "ProfitFactor": None
                },
                "FineTuneInfo": {
                    "NumSamples": len(X_tune),
                    "AvgValSharpe": float(avg_val_sharpe)
                },
                "Message": f"Fine-tuning complete with {len(X_tune)} samples"
            }
        
        else:  # predict mode
            eprint("\n--- PREDICTION MODE ---")
            
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Model not found: {args.model_path}")
            
            # Load model
            from tensorflow.keras.models import load_model
            model = load_model(args.model_path)
            eprint(f"[OK] Loaded model from: {args.model_path}")
            
            # Extract features from last 60 days
            features = extract_features(df)
            if len(features) < args.lookback:
                raise ValueError(f"Insufficient data for prediction: need {args.lookback}, got {len(features)}")
            
            features_60d = features.iloc[-args.lookback:].values
            
            # Predict parameters
            predicted_params = predict_parameters(model, features_60d, PARAM_GRID)
            eprint(f"[OK] Predicted parameters: {predicted_params}")
            
            # Calculate baseline parameters
            baseline_params = calculate_baseline_params(PARAM_GRID)
            
            # Quick validation on last 30 days
            if len(df) >= 30:
                validation_df = df.iloc[-30:].copy()
                
                # Test predicted parameters
                pred_params_dict = {
                    'ma_period': predicted_params['MovingAveragePeriod'],
                    'std_dev_multiplier': predicted_params['StdDevMultiplier'],
                    'rsi_period': predicted_params['RSIPeriod'],
                    'rsi_oversold': predicted_params['RSIOversold'],
                    'rsi_overbought': predicted_params['RSIOverbought'],
                    'atr_period': predicted_params['AtrPeriod'],
                    'atr_multiplier': predicted_params['AtrMultiplier']
                }
                _, pred_metrics = optimize_parameters(
                    validation_df,
                    {k: [v] for k, v in pred_params_dict.items()}
                )
                
                # Test baseline parameters
                baseline_params_dict = {
                    'ma_period': baseline_params['MovingAveragePeriod'],
                    'std_dev_multiplier': baseline_params['StdDevMultiplier'],
                    'rsi_period': baseline_params['RSIPeriod'],
                    'rsi_oversold': baseline_params['RSIOversold'],
                    'rsi_overbought': baseline_params['RSIOverbought'],
                    'atr_period': baseline_params['AtrPeriod'],
                    'atr_multiplier': baseline_params['AtrMultiplier']
                }
                _, baseline_metrics = optimize_parameters(
                    validation_df,
                    {k: [v] for k, v in baseline_params_dict.items()}
                )
                
                # Calculate alpha
                alpha = pred_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']
                improvement_pct = (alpha / baseline_metrics['sharpe_ratio'] * 100) if baseline_metrics['sharpe_ratio'] != 0 else 0
                
                eprint(f"\n{'='*60}")
                eprint(f"Validation Results (last 30 days):")
                eprint(f"  Predicted Sharpe: {pred_metrics['sharpe_ratio']:.4f}")
                eprint(f"  Baseline Sharpe:  {baseline_metrics['sharpe_ratio']:.4f}")
                eprint(f"  Alpha:            {alpha:.4f} ({improvement_pct:+.1f}%)")
                eprint(f"{'='*60}\n")
                
                result = {
                    "Status": "success",
                    "BestParameters": predicted_params,
                    "Metrics": {
                        "TotalReturns": pred_metrics['total_return'],
                        "SharpeRatio": pred_metrics['sharpe_ratio'],
                        "WinRate": pred_metrics['win_rate'],
                        "NumTrades": pred_metrics['num_trades'],
                        "MaxDrawdown": pred_metrics['max_drawdown'],
                        "ProfitFactor": pred_metrics['profit_factor']
                    },
                    "BaselineComparison": {
                        "BaselineParameters": baseline_params,
                        "BaselineSharpe": baseline_metrics['sharpe_ratio'],
                        "Alpha": alpha,
                        "ImprovementPercent": improvement_pct
                    },
                    "Message": "Prediction complete with validation"
                }
            else:
                result = {
                    "Status": "success",
                    "BestParameters": predicted_params,
                    "Metrics": {
                        "TotalReturns": None,
                        "SharpeRatio": None,
                        "WinRate": None,
                        "NumTrades": None,
                        "MaxDrawdown": None,
                        "ProfitFactor": None
                    },
                    "Message": "Prediction complete (insufficient data for validation)"
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
    parser = argparse.ArgumentParser(description="Deep Learning Regime-Based Parameter Optimizer")
    parser.add_argument("input_csv_path", help="Path to input CSV file")
    parser.add_argument("output_json_path", help="Path to output JSON file")
    parser.add_argument("--mode", required=True, choices=['train', 'fine-tune', 'predict'],
                       help="Operation mode")
    parser.add_argument("--model-path", required=True, help="Path to model file (.keras)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback window size")
    
    args = parser.parse_args()
    main(args)