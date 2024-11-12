# 1. IMPORT STATEMENTS
import pandas as pd  
# Pandas is used for data manipulation and analysis
# pd is the conventional alias for pandas
# We use it here to read CSV files and handle structured data

import numpy as np
# NumPy is fundamental for numerical computations
# Provides support for large, multi-dimensional arrays and matrices
# Also provides mathematical functions to operate on these arrays

from sklearn.preprocessing import StandardScaler
# StandardScaler is used for feature normalization
# Transforms features to have mean=0 and variance=1
# Essential for neural networks to work effectively

from tensorflow.keras.models import Sequential
# Sequential is the simplest type of Keras model
# Allows us to build neural networks layer by layer
# Used for both our LSTM and CNN models

from tensorflow.keras.layers import *
# Imports all layer types from Keras
# Includes LSTM, Dense, Conv1D, Flatten layers
# '*' means import everything (not best practice in production code)

from tensorflow.keras.callbacks import ModelCheckpoint
# ModelCheckpoint allows saving model weights during training
# Can save best model based on validation metrics
# Useful for model persistence and preventing loss of progress

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# These are evaluation metrics for regression problems
# mean_squared_error: Average of squared prediction errors
# mean_absolute_error: Average of absolute prediction errors
# r2_score: Proportion of variance explained by model

import pickle
# Pickle is used for serializing Python objects
# Allows saving trained models and scalers to disk
# Can load them later for making predictions

# 2. EVALUATION FUNCTION
def evaluate_model(y_true, y_pred, model_name):
    """
    Calculate and display accuracy metrics for regression models
    
    Parameters:
    y_true: Actual values
    y_pred: Predicted values
    model_name: Name of the model being evaluated
    
    Returns:
    Dictionary containing the calculated metrics
    """
    # Calculate MSE - sensitive to outliers due to squaring
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate MAE - linear scale, more robust to outliers
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate R² - indicates goodness of fit (0 to 1)
    r2 = r2_score(y_true, y_pred)
    
    # Print formatted results
    print(f"\n{model_name} Performance Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Return metrics dictionary for later use
    return {'mse': mse, 'mae': mae, 'r2': r2}

# 3. DATA PREPARATION FUNCTION
def prepare_data(seq, num):
    """
    Create sequences for time series prediction
    
    Parameters:
    seq: Original time series data
    num: Number of time steps to use for prediction
    
    Returns:
    Arrays of input sequences and target values
    """
    x, y = [], []  # Initialize empty lists
    
    # Create sequences with sliding window
    for i in range(0, (len(seq)-num), 1):
        # Input sequence of length 'num'
        x.append(seq[i:i+num])
        # Target is the next value after sequence
        y.append(seq[i+num])
    
    # Convert lists to numpy arrays for ML processing
    return np.array(x), np.array(y)

# 4. DATA LOADING AND PREPROCESSING
# Load data from CSV file
data = pd.read_csv('webtraffic.csv')

# Extract sessions column as numpy array
sessions = data['Sessions'].values

# Define sequence length (e.g., 168 hours = 1 week)
num = 168

# Create sequences for training
x, y = prepare_data(sessions, num)

# 5. DATA SPLITTING
# Calculate split index (90% training, 10% validation)
ind = int(0.9 * len(x))

# Split data into training and validation sets
x_tr, y_tr = x[:ind], y[:ind]  # Training data
x_val, y_val = x[ind:], y[ind:]  # Validation data

# 6. DATA SCALING
# Initialize scalers
x_scaler = StandardScaler()
y_scaler = StandardScaler()

# Scale input features
x_tr = x_scaler.fit_transform(x_tr)  # Fit and transform training data
x_val = x_scaler.transform(x_val)    # Transform validation data only

# Reshape and scale target values
y_tr = y_tr.reshape(len(y_tr), 1)    # Reshape for scaler
y_val = y_val.reshape(len(y_val), 1)
y_tr = y_scaler.fit_transform(y_tr)[:, 0]  # Fit and transform
y_val = y_scaler.transform(y_val)[:, 0]    # Transform only

# Reshape data for 3D input (samples, time steps, features)
x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

# 7. LSTM MODEL
print("\nTraining LSTM Model...")
lstm_model = Sequential([
    # LSTM layer with 128 units
    # input_shape=(168, 1) means 168 time steps, 1 feature
    LSTM(128, input_shape=(168, 1)),
    
    # Dense hidden layer with 64 units and ReLU activation
    Dense(64, activation='relu'),
    
    # Output layer with 1 unit (prediction)
    Dense(1, activation='linear')
])

# Compile model with MSE loss and Adam optimizer
lstm_model.compile(loss='mse', optimizer='adam')

# Train LSTM model
lstm_history = lstm_model.fit(
    x_tr, y_tr,                # Training data
    epochs=30,                 # Number of training cycles
    batch_size=32,            # Samples per gradient update
    validation_data=(x_val, y_val),  # Validation data
    verbose=1                 # Show progress bar
)

# 8. EVALUATE LSTM
# Make predictions
lstm_pred = lstm_model.predict(x_val)

# Inverse transform predictions and actual values
lstm_pred = y_scaler.inverse_transform(lstm_pred)
y_val_orig = y_scaler.inverse_transform(y_val.reshape(-1, 1))

# Calculate metrics
lstm_metrics = evaluate_model(y_val_orig, lstm_pred, "LSTM")

# 9. CNN MODEL
print("\nTraining CNN Model...")
cnn_model = Sequential([
    # First Conv1D layer
    # 64 filters, kernel size 3, same padding
    Conv1D(64, 3, padding='same', activation='relu', input_shape=(num, 1)),
    
    # Second Conv1D layer
    # 32 filters, kernel size 5
    Conv1D(32, 5, padding='same', activation='relu'),
    
    # Flatten layer to convert 3D to 2D
    Flatten(),
    
    # Dense hidden layer
    Dense(64, activation='relu'),
    
    # Output layer
    Dense(1, activation='linear')
])

# Compile CNN model
cnn_model.compile(loss='mse', optimizer='adam')

# Train CNN model
cnn_history = cnn_model.fit(
    x_tr, y_tr,
    epochs=30,
    batch_size=32,
    validation_data=(x_val, y_val),
    verbose=1
)

# 10. EVALUATE CNN
# Make predictions
cnn_pred = cnn_model.predict(x_val)

# Inverse transform predictions
cnn_pred = y_scaler.inverse_transform(cnn_pred)

# Calculate metrics
cnn_metrics = evaluate_model(y_val_orig, cnn_pred, "CNN")

# 11. SAVE MODELS AND DATA
# Create dictionary of all objects to save
model_data = {
    'lstm_model': lstm_model,      # LSTM model
    'cnn_model': cnn_model,        # CNN model
    'x_scaler': x_scaler,          # Input scaler
    'y_scaler': y_scaler,          # Output scaler
    'lstm_metrics': lstm_metrics,  # LSTM performance metrics
    'cnn_metrics': cnn_metrics     # CNN performance metrics
}

# Save everything to file
with open('traffic_models.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\nModels and metrics saved successfully!")
