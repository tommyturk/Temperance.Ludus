# ---- (Pre-train & Fine-tune) ----
import argparse
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

MODEL_DIR = "/app/models"

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def build_model(look_back):
    """Builds the LSTM model architecture."""
    model = Sequential([
        LSTM(100, input_shape=(look_back, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    parser = argparse.ArgumentParser(description="Ludus LSTM Optimizer")
    parser.add_argument("--mode", type=str, choices=['pretrain', 'finetune'], required=True, help="Operating mode")
    parser.add_argument("--symbol", type=str, required=True, help="The stock symbol (e.g., NVDA)")
    parser.add_argument("--interval", type=str, required=True, help="The time interval (e.g., 1d)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the historical data CSV")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--look_back", type=int, default=60, help="Number of previous time steps for input")
    args = parser.parse_args()

    model_filename = os.path.join(MODEL_DIR, f"{args.symbol}_{args.interval}_base_model.keras")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- GPU Verification ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow has detected {len(gpus)} GPU(s).")
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU detected. Running on CPU.")

    # --- Data Loading and Scaling ---
    dataframe = pd.read_csv(args.data_path)
    dataset = dataframe['ClosePrice'].values.reshape(-1, 1).astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_dataset = scaler.fit_transform(dataset)
    
    # --- Mode-Specific Logic ---
    if args.mode == 'pretrain':
        print(f"--- Running in PRE-TRAIN mode for {args.symbol} ---")
        trainX, trainY = create_dataset(scaled_dataset, args.look_back)
        if len(trainX) == 0:
            print(f"Error: Not enough data for pre-training. Need > {args.look_back + 1} points.")
            exit(1)
        
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        
        model = build_model(args.look_back)
        print(f"Starting pre-training for {args.epochs} epochs...")
        model.fit(trainX, trainY, epochs=args.epochs, batch_size=32, verbose=0)
        
        # Save the newly trained base model
        model.save(model_filename)
        print(f"Pre-training complete. Model saved to {model_filename}")
        status_result = { "Status": "Pre-training complete", "ModelFile": model_filename }
        print(json.dumps(status_result))
        return

    elif args.mode == 'finetune':
        print(f"--- Running in FINE-TUNE mode for {args.symbol} ---")
        if not os.path.exists(model_filename):
            print(f"Error: No pre-trained model found at {model_filename}. Please run pre-training first.")
            exit(1)

        # Load the foundational model
        model = load_model(model_filename)
        print(f"Loaded pre-trained model from {model_filename}")

        # Fine-tune on the recent data
        trainX, trainY = create_dataset(scaled_dataset, args.look_back)
        if len(trainX) == 0:
            print(f"Error: Not enough recent data for fine-tuning. Need > {args.look_back + 1} points.")
            exit(1)

        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

        print(f"Starting fine-tuning for {args.epochs} epochs...")
        model.fit(trainX, trainY, epochs=args.epochs, batch_size=32, verbose=0)
        
        # Save the updated model over the old one, so knowledge accumulates
        model.save(model_filename)
        print(f"Fine-tuning complete. Updated model saved to {model_filename}")

        # --- "Optimization" Logic ---
        last_sequence_scaled = scaled_dataset[-args.look_back:]
        last_sequence_scaled = np.reshape(last_sequence_scaled, (1, args.look_back, 1))
        predicted_price_scaled = model.predict(last_sequence_scaled, verbose=0)
        
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        last_actual_price = dataset[-1][0]

        # CPO Logic
        ma_period = 20
        rsi_oversold = 30
        if predicted_price[0][0] > last_actual_price * 1.01:
            ma_period = 15
            rsi_oversold = 35
        
        optimized_params = {
            "PredictedNextClose": float(predicted_price[0][0]),
            "LastActualClose": float(last_actual_price),
            "OptimizedParameters": {
                "MovingAveragePeriod": ma_period,
                "RSIOversold": rsi_oversold,
                "RSIOverbought": 70
            }
        }
        
        # Output the final parameters
        print(json.dumps(optimized_params))

if __name__ == "__main__":
    main()