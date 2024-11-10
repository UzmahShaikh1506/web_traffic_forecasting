import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and display accuracy metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {'mse': mse, 'mae': mae, 'r2': r2}

def prepare_data(seq, num):
    x, y = [], []
    for i in range(0, (len(seq)-num), 1):
        x.append(seq[i:i+num])
        y.append(seq[i+num])
    return np.array(x), np.array(y)

# Load and preprocess data
data = pd.read_csv('webtraffic.csv')
sessions = data['Sessions'].values
num = 168
x, y = prepare_data(sessions, num)

# Split data
ind = int(0.9 * len(x))
x_tr, y_tr = x[:ind], y[:ind]
x_val, y_val = x[ind:], y[ind:]

# Scale data
x_scaler = StandardScaler()
x_tr = x_scaler.fit_transform(x_tr)
x_val = x_scaler.transform(x_val)

y_tr = y_tr.reshape(len(y_tr), 1)
y_val = y_val.reshape(len(y_val), 1)
y_scaler = StandardScaler()
y_tr = y_scaler.fit_transform(y_tr)[:, 0]
y_val = y_scaler.transform(y_val)[:, 0]

# Reshape for models
x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

# LSTM Model
print("\nTraining LSTM Model...")
lstm_model = Sequential([
    LSTM(128, input_shape=(168, 1)),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])
lstm_model.compile(loss='mse', optimizer='adam')
lstm_history = lstm_model.fit(x_tr, y_tr, epochs=30, batch_size=32, 
                            validation_data=(x_val, y_val), verbose=1)

# Evaluate LSTM
lstm_pred = lstm_model.predict(x_val)
lstm_pred = y_scaler.inverse_transform(lstm_pred)
y_val_orig = y_scaler.inverse_transform(y_val.reshape(-1, 1))
lstm_metrics = evaluate_model(y_val_orig, lstm_pred, "LSTM")

# CNN Model
print("\nTraining CNN Model...")
cnn_model = Sequential([
    Conv1D(64, 3, padding='same', activation='relu', input_shape=(num, 1)),
    Conv1D(32, 5, padding='same', activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])
cnn_model.compile(loss='mse', optimizer='adam')
cnn_history = cnn_model.fit(x_tr, y_tr, epochs=30, batch_size=32, 
                          validation_data=(x_val, y_val), verbose=1)

# Evaluate CNN
cnn_pred = cnn_model.predict(x_val)
cnn_pred = y_scaler.inverse_transform(cnn_pred)
cnn_metrics = evaluate_model(y_val_orig, cnn_pred, "CNN")

# Save models, scalers, and metrics
model_data = {
    'lstm_model': lstm_model,
    'cnn_model': cnn_model,
    'x_scaler': x_scaler,
    'y_scaler': y_scaler,
    'lstm_metrics': lstm_metrics,
    'cnn_metrics': cnn_metrics
}

with open('traffic_models.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\nModels and metrics saved successfully!")