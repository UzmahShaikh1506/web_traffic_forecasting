# Deep Learning Code Analysis: Web Traffic Prediction

## 1. Import Statements

### Pandas Import
```python
import pandas as pd
```
* **Purpose**: Data manipulation and analysis
* **Usage**: Reading CSV files and handling structured data
* **Why pd**: Standard convention for pandas alias

### NumPy Import
```python
import numpy as np
```
* **Purpose**: Numerical computations
* **Features**: 
  * Support for large multi-dimensional arrays
  * Mathematical functions for array operations
  * Essential for machine learning operations

### StandardScaler Import
```python
from sklearn.preprocessing import StandardScaler
```
* **Purpose**: Feature normalization
* **What it does**: 
  * Transforms features to mean=0 and variance=1
  * Critical for neural network performance
  * Ensures all features are on similar scales

### Keras Sequential Import
```python
from tensorflow.keras.models import Sequential
```
* **Purpose**: Neural network model construction
* **Why Sequential**: 
  * Simplest type of Keras model
  * Linear stack of layers
  * Suitable for most deep learning tasks

### Keras Layers Import
```python
from tensorflow.keras.layers import *
```
* **Purpose**: Access to all neural network layer types
* **Includes**: 
  * LSTM layers
  * Dense layers
  * Conv1D layers
  * Flatten layers
* **Note**: '*' imports everything (not recommended for production)

### ModelCheckpoint Import
```python
from tensorflow.keras.callbacks import ModelCheckpoint
```
* **Purpose**: Model saving during training
* **Benefits**:
  * Save best model based on metrics
  * Prevent progress loss
  * Enable model persistence

### Metrics Import
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```
* **Purpose**: Model evaluation metrics
* **Metrics included**:
  * MSE: Average of squared errors
  * MAE: Average of absolute errors
  * R²: Variance explanation ratio

### Pickle Import
```python
import pickle
```
* **Purpose**: Object serialization
* **Uses**:
  * Save trained models
  * Save scalers
  * Enable model reuse

## 2. Evaluation Function

```python
def evaluate_model(y_true, y_pred, model_name):
```
* **Purpose**: Calculate and display model performance metrics
* **Parameters**:
  * `y_true`: Actual values
  * `y_pred`: Model predictions
  * `model_name`: Identifier for the model
* **Returns**: Dictionary of metrics
* **Metrics calculated**:
  * MSE (sensitive to outliers)
  * MAE (robust to outliers)
  * R² (goodness of fit)

## 3. Data Preparation Function

```python
def prepare_data(seq, num):
```
* **Purpose**: Create sequences for time series prediction
* **Method**: Sliding window approach
* **Parameters**:
  * `seq`: Original time series
  * `num`: Sequence length
* **Returns**: Arrays for input (X) and target (y)
* **Example**:
  * Input: `[1,2,3,4,5]`, `num=2`
  * X: `[[1,2], [2,3], [3,4]]`
  * y: `[3, 4, 5]`

## 4. Data Loading and Preprocessing

```python
data = pd.read_csv('webtraffic.csv')
sessions = data['Sessions'].values
num = 168
```
* **Steps**:
  1. Load CSV data
  2. Extract sessions column
  3. Set sequence length (168 hours = 1 week)
* **Why 168**: Captures weekly patterns in hourly data

## 5. Data Splitting

```python
ind = int(0.9 * len(x))
x_tr, y_tr = x[:ind], y[:ind]
x_val, y_val = x[ind:], y[ind:]
```
* **Split ratio**: 90% training, 10% validation
* **Purpose**: 
  * Training data: Model learning
  * Validation data: Performance evaluation

## 6. Data Scaling

```python
x_scaler = StandardScaler()
y_scaler = StandardScaler()
```
* **Process**:
  1. Initialize scalers
  2. Fit and transform training data
  3. Transform validation data
* **Why scale**:
  * Neural networks perform better with normalized data
  * Prevents dominance of large-scale features
  * Improves convergence

## 7. LSTM Model

```python
lstm_model = Sequential([
    LSTM(128, input_shape=(168, 1)),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])
```
* **Architecture**:
  * LSTM layer: 128 units
  * Dense hidden layer: 64 units, ReLU
  * Output layer: 1 unit, linear
* **Why this structure**:
  * LSTM: Good for time series
  * Dense layers: Pattern refinement
  * Linear output: Regression task

## 8. CNN Model

```python
cnn_model = Sequential([
    Conv1D(64, 3, padding='same', activation='relu', input_shape=(num, 1)),
    Conv1D(32, 5, padding='same', activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])
```
* **Architecture**:
  * First Conv1D: 64 filters, size 3
  * Second Conv1D: 32 filters, size 5
  * Flatten layer
  * Dense layers: Similar to LSTM
* **Design choices**:
  * Multiple kernel sizes: Capture different patterns
  * Decreasing filters: Reduce complexity
  * Same padding: Preserve sequence length

## 9. Model Training

```python
model.fit(x_tr, y_tr, epochs=30, batch_size=32, validation_data=(x_val, y_val))
```
* **Parameters**:
  * epochs=30: Training cycles
  * batch_size=32: Samples per update
  * validation_data: Performance monitoring
* **Why these values**:
  * 30 epochs: Good starting point
  * 32 batch size: Memory/performance trade-off

## 10. Model Saving

```python
with open('traffic_models.pkl', 'wb') as f:
    pickle.dump(model_data, f)
```
* **Saved items**:
  * Both models (LSTM & CNN)
  * Scalers
  * Performance metrics
* **Purpose**: Enable model reuse and comparison

## Key Design Decisions

### Why Two Models?
1. **LSTM Benefits**:
   * Captures long-term patterns
   * Designed for sequential data
   * Remembers important patterns

2. **CNN Benefits**:
   * Faster training
   * Captures local patterns
   * Less memory intensive

### Training Choices
* **Window Size**: 168 (one week of hourly data)
* **Split Ratio**: 90/10 (standard practice)
* **Scaling**: Essential for neural networks
* **Batch Size**: 32 (common choice)
* **Epochs**: 30 (reasonable starting point)

### Architecture Decisions
* **LSTM**:
  * Single layer (often sufficient)
  * 128 units (good capacity)
  * Dense layer for refinement

* **CNN**:
  * Two conv layers (hierarchical patterns)
  * Different kernel sizes (3,5)
  * Same padding (maintain length)
