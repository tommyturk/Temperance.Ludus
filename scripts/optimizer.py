# scripts/optimizer.py
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_dataset(dataset, look_back=1):
    """Create input and output sequences for the LSTM."""
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def main():
    parser = argparse.ArgumentParser(description="Ludus LSTM Optimizer")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy name (for context)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the historical data CSV")
    args = parser.parse_args()

    # --- 1. GPU Verification ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow has detected {len(gpus)} GPU(s). Using GPU.")
        try:
            # Set memory growth to avoid allocating all memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"RuntimeError in GPU setup: {e}")
    else:
        print("No GPU detected by TensorFlow. Running on CPU.")

    # --- 2. Data Loading and Preparation ---
    try:
        dataframe = pd.read_csv(args.data_path)
        # Use only the 'ClosePrice' for this model
        dataset = dataframe['ClosePrice'].values.astype('float32')
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        exit(1)

    # Reshape and create sequences
    dataset = np.reshape(dataset, (-1, 1))
    look_back = 20  # Use the last 20 days to predict the next
    trainX, trainY = create_dataset(dataset, look_back)
    
    # Reshape input to be [samples, time steps, features] as required by LSTM
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    # --- 3. LSTM Model Definition ---
    model = Sequential([
        LSTM(50, input_shape=(look_back, 1), return_sequences=True), # 50 units, return sequence for next LSTM layer
        LSTM(50), # Second LSTM layer
        Dense(1)  # Output layer with 1 neuron for the predicted price
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # --- 4. Model Training ---
    print(f"Starting model training for strategy '{args.strategy}'...")
    # In a real scenario, epochs would be a configurable parameter
    model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=0)
    print("Model training completed.")

    # --- 5. "Optimization" Logic ---
    # For this example, we'll use the model to predict the next price
    # and adjust a parameter based on that prediction. This simulates CPO.
    last_sequence = np.array([dataset[-look_back:, 0]])
    last_sequence = np.reshape(last_sequence, (last_sequence.shape[0], last_sequence.shape[1], 1))
    predicted_price = model.predict(last_sequence, verbose=0)
    last_actual_price = dataset[-1][0]

    # Dummy logic: if predicted price is higher, use a shorter MA period.
    ma_period = 20
    if predicted_price[0][0] > last_actual_price:
        ma_period = 15 # Shorter MA for expected upward trend

    # --- 6. Output Results as JSON ---
    optimized_params = {
        "Note": "This is a simulated optimization result from the LSTM model.",
        "PredictedNextClose": float(predicted_price[0][0]),
        "LastActualClose": float(last_actual_price),
        "OptimizedMAPeriod": ma_period,
        "OriginalRSIPeriod": 14 # Keep other params static for now
    }
    
    # Print the final JSON result to stdout for the C# process to capture
    print(json.dumps(optimized_params))

if __name__ == "__main__":
    main()