# ---- (Pre-train & Fine-tune with GPU Support) ----
import argparse
import json
import os
import numpy as np
import pandas as pd
import sys
import traceback

# Configure TensorFlow for optimal GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Show GPU info but reduce other warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

MODEL_DIR = "/app/models"

def configure_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Optionally set memory limit (uncomment if needed)
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpus[0],
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)]  # 4GB limit
            # )
            
            print(f"GPU configuration successful. Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        except RuntimeError as e:
            print(f"GPU configuration failed: {e}")
            return False
    return len(gpus) > 0

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def build_model(look_back, use_gpu=True):
    """Builds the LSTM model architecture optimized for GPU"""
    if use_gpu:
        # Larger model for GPU - can handle more complexity
        model = Sequential([
            LSTM(128, input_shape=(look_back, 1), return_sequences=True),
            Dropout(0.3),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(50),
            Dense(25),
            Dense(1)
        ])
    else:
        # Smaller model for CPU fallback
        model = Sequential([
            LSTM(50, input_shape=(look_back, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
    
    # Use mixed precision for better GPU performance
    if use_gpu:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',  # More robust loss function
            metrics=['mae']
        )
    else:
        model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def main():
    try:
        parser = argparse.ArgumentParser(description="Ludus LSTM Optimizer with GPU Support")
        parser.add_argument("--mode", type=str, choices=['pretrain', 'finetune'], required=True, help="Operating mode")
        parser.add_argument("--symbol", type=str, required=True, help="The stock symbol (e.g., NVDA)")
        parser.add_argument("--interval", type=str, required=True, help="The time interval (e.g., 1d)")
        parser.add_argument("--data_path", type=str, required=True, help="Path to the historical data CSV")
        parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
        parser.add_argument("--look_back", type=int, default=60, help="Number of previous time steps for input")
        parser.add_argument("--force_cpu", action='store_true', help="Force CPU usage even if GPU is available")
        args = parser.parse_args()

        print(f"Starting Ludus optimizer with mode: {args.mode}")
        print(f"Data path: {args.data_path}")
        print(f"Symbol: {args.symbol}, Interval: {args.interval}")
        print(f"Epochs: {args.epochs}, Look back: {args.look_back}")

        model_filename = os.path.join(MODEL_DIR, f"{args.symbol}_{args.interval}_base_model.keras")
        os.makedirs(MODEL_DIR, exist_ok=True)

        # --- GPU/CPU Configuration ---
        print(f"TensorFlow version: {tf.__version__}")
        
        use_gpu = False
        if not args.force_cpu:
            use_gpu = configure_gpu()
        
        if use_gpu:
            print("? GPU mode enabled - using optimized GPU configuration")
            # Enable mixed precision for better performance
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        else:
            print("? CPU mode - GPU not available or forced CPU usage")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Print available devices
        print("Available devices:")
        for device in tf.config.list_physical_devices():
            print(f"  - {device.device_type}: {device.name}")

        # --- Data Loading and Validation ---
        print(f"Loading data from: {args.data_path}")
        
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data file not found: {args.data_path}")
        
        dataframe = pd.read_csv(args.data_path)
        print(f"Loaded {len(dataframe)} rows of data")
        print(f"Columns: {list(dataframe.columns)}")
        
        if 'ClosePrice' not in dataframe.columns:
            raise ValueError(f"'ClosePrice' column not found in data. Available columns: {list(dataframe.columns)}")
        
        # Check for missing values
        if dataframe['ClosePrice'].isna().any():
            print("Warning: Found missing values in ClosePrice, removing them...")
            dataframe = dataframe.dropna(subset=['ClosePrice'])
            print(f"After removing missing values: {len(dataframe)} rows")
        
        if len(dataframe) < args.look_back + 2:
            raise ValueError(f"Insufficient data: need at least {args.look_back + 2} rows, got {len(dataframe)}")
        
        dataset = dataframe['ClosePrice'].values.reshape(-1, 1).astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_dataset = scaler.fit_transform(dataset)
        
        print(f"Data shape after scaling: {scaled_dataset.shape}")
        
        # --- Mode-Specific Logic ---
        if args.mode == 'pretrain':
            print(f"--- Running in PRE-TRAIN mode for {args.symbol} ---")
            trainX, trainY = create_dataset(scaled_dataset, args.look_back)
            if len(trainX) == 0:
                raise ValueError(f"Error: Not enough data for pre-training. Need > {args.look_back + 1} points.")
            
            trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
            print(f"Training data shape: X={trainX.shape}, Y={trainY.shape}")
            
            model = build_model(args.look_back, use_gpu)
            print(f"Model created with {'GPU' if use_gpu else 'CPU'} optimizations")
            
            # Adjust epochs and batch size based on GPU availability
            actual_epochs = args.epochs if use_gpu else min(args.epochs, 10)
            batch_size = 64 if use_gpu else 16
            
            print(f"Starting pre-training for {actual_epochs} epochs with batch size {batch_size}")
            
            # Add callbacks for better training
            callbacks = []
            if use_gpu:
                # Early stopping to prevent overfitting
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True
                ))
                # Reduce learning rate when loss plateaus
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3
                ))
            
            with tf.device('/GPU:0' if use_gpu else '/CPU:0'):
                history = model.fit(
                    trainX, trainY, 
                    epochs=actual_epochs, 
                    batch_size=batch_size, 
                    verbose=1, 
                    validation_split=0.2,
                    callbacks=callbacks
                )
            
            # Save the newly trained base model
            model.save(model_filename)
            print(f"Pre-training complete. Model saved to {model_filename}")
            
            status_result = { 
                "Status": "Pre-training complete", 
                "ModelFile": model_filename,
                "FinalLoss": float(history.history['loss'][-1]),
                "UsedGPU": use_gpu,
                "Epochs": actual_epochs
            }
            print(json.dumps(status_result))
            return

        elif args.mode == 'finetune':
            print(f"--- Running in FINE-TUNE mode for {args.symbol} ---")
            if not os.path.exists(model_filename):
                print(f"No pre-trained model found at {model_filename}. Creating new model...")
                model = build_model(args.look_back, use_gpu)
            else:
                # Load the foundational model
                try:
                    model = load_model(model_filename)
                    print(f"Loaded pre-trained model from {model_filename}")
                except Exception as e:
                    print(f"Failed to load model: {e}. Creating new model...")
                    model = build_model(args.look_back, use_gpu)

            # Fine-tune on the recent data
            trainX, trainY = create_dataset(scaled_dataset, args.look_back)
            if len(trainX) == 0:
                raise ValueError(f"Error: Not enough recent data for fine-tuning. Need > {args.look_back + 1} points.")

            trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
            print(f"Fine-tuning data shape: X={trainX.shape}, Y={trainY.shape}")

            # Adjust parameters for fine-tuning
            actual_epochs = min(args.epochs, 20) if use_gpu else min(args.epochs, 5)
            batch_size = 32 if use_gpu else 16
            
            print(f"Starting fine-tuning for {actual_epochs} epochs with batch size {batch_size}")
            
            # Lower learning rate for fine-tuning
            if use_gpu:
                model.optimizer.learning_rate = 0.0001
            
            callbacks = []
            if use_gpu:
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=3, restore_best_weights=True
                ))
            
            with tf.device('/GPU:0' if use_gpu else '/CPU:0'):
                history = model.fit(
                    trainX, trainY, 
                    epochs=actual_epochs, 
                    batch_size=batch_size, 
                    verbose=1, 
                    validation_split=0.2,
                    callbacks=callbacks
                )
            
            # Save the updated model
            model.save(model_filename)
            print(f"Fine-tuning complete. Updated model saved to {model_filename}")

            # --- "Optimization" Logic ---
            if len(scaled_dataset) >= args.look_back:
                last_sequence_scaled = scaled_dataset[-args.look_back:]
                last_sequence_scaled = np.reshape(last_sequence_scaled, (1, args.look_back, 1))
                
                with tf.device('/GPU:0' if use_gpu else '/CPU:0'):
                    predicted_price_scaled = model.predict(last_sequence_scaled, verbose=0)
                
                predicted_price = scaler.inverse_transform(predicted_price_scaled)
                last_actual_price = dataset[-1][0]

                # Enhanced CPO Logic with more sophisticated parameters
                ma_period = 20
                rsi_oversold = 30
                rsi_overbought = 70
                
                price_change_pct = (predicted_price[0][0] - last_actual_price) / last_actual_price
                
                if price_change_pct > 0.02:  # Strong upward prediction
                    ma_period = 10
                    rsi_oversold = 40
                    rsi_overbought = 75
                elif price_change_pct > 0.01:  # Moderate upward prediction
                    ma_period = 15
                    rsi_oversold = 35
                    rsi_overbought = 72
                elif price_change_pct < -0.02:  # Strong downward prediction
                    ma_period = 30
                    rsi_oversold = 25
                    rsi_overbought = 65
                elif price_change_pct < -0.01:  # Moderate downward prediction
                    ma_period = 25
                    rsi_oversold = 28
                    rsi_overbought = 68
                
                optimized_params = {
                    "PredictedNextClose": float(predicted_price[0][0]),
                    "LastActualClose": float(last_actual_price),
                    "PredictedChangePercent": float(price_change_pct * 100),
                    "OptimizedParameters": {
                        "MovingAveragePeriod": ma_period,
                        "RSIOversold": rsi_oversold,
                        "RSIOverbought": rsi_overbought,
                        "VolatilityAdjustment": abs(price_change_pct) > 0.015
                    },
                    "ModelLoss": float(history.history['loss'][-1]) if 'loss' in history.history else None,
                    "UsedGPU": use_gpu,
                    "TrainingEpochs": actual_epochs
                }
            else:
                # Fallback if not enough data for prediction
                optimized_params = {
                    "Error": "Insufficient data for prediction",
                    "OptimizedParameters": {
                        "MovingAveragePeriod": 20,
                        "RSIOversold": 30,
                        "RSIOverbought": 70,
                        "VolatilityAdjustment": False
                    },
                    "UsedGPU": use_gpu
                }
            
            # Output the final parameters
            print(json.dumps(optimized_params))

    except Exception as e:
        error_details = {
            "Error": str(e),
            "Type": type(e).__name__,
            "Traceback": traceback.format_exc()
        }
        print(f"PYTHON_ERROR: {json.dumps(error_details)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()