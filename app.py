import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px

# Page config
st.set_page_config(page_title="Web Traffic Forecaster", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .css-1d391kg {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        color: #000;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📊 Web Traffic Forecaster")
    st.markdown("---")
    st.markdown("""
    ### Instructions:
    1. Upload your CSV file with web traffic data
    2. Select your preferred model (CNN or LSTM)
    3. View predictions, accuracy metrics, and download results
    
    ### Data Format:
    - CSV file with 'Sessions' column
    - Hourly web traffic data
    """)

def load_models():
    try:
        with open('traffic_models.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def prepare_data(seq, num):
    x = []
    for i in range(0, (len(seq)-num), 1):
        x.append(seq[i:i+num])
    return np.array(x)

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

def forecast(model, x_val, no_of_pred, ind, x_scaler, y_scaler):
    predictions = []
    temp = x_val[ind]
    
    for i in range(no_of_pred):
        pred = model.predict(temp.reshape(1, -1, 1), verbose=0)[0][0]
        temp = np.insert(temp, len(temp), pred)
        predictions.append(pred)
        temp = temp[1:]
    
    predictions = np.array(predictions).reshape(-1, 1)
    return y_scaler.inverse_transform(predictions)

# Main app
st.title("Web Traffic Forecasting")

# Model selection
model_choice = st.selectbox(
    "Select Model",
    ["CNN", "LSTM"],
    help="Choose between CNN or LSTM model for prediction"
)

# Display model metrics
model_data = load_models()
if model_data:
    metrics = model_data['cnn_metrics'] if model_choice == "CNN" else model_data['lstm_metrics']
    
    st.markdown("### Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Mean Squared Error</h4>
                <h2>{metrics['mse']:.4f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Mean Absolute Error</h4>
                <h2>{metrics['mae']:.4f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>R² Score</h4>
                <h2>{metrics['r2']:.4f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data and models
        data = pd.read_csv(uploaded_file)
        
        if model_data is None:
            st.error("Failed to load models. Please check if the model file exists.")
            st.stop()
        
        with st.spinner("Processing data..."):
            # Prepare data
            sessions = data['Sessions'].values
            num = 168
            x = prepare_data(sessions, num)
            x_scaled = model_data['x_scaler'].transform(x)
            x_scaled = x_scaled.reshape(x_scaled.shape[0], x_scaled.shape[1], 1)
            
            # Make predictions
            selected_model = model_data['cnn_model'] if model_choice == "CNN" else model_data['lstm_model']
            predictions = forecast(selected_model, x_scaled, 24, 0,
                                model_data['x_scaler'],
                                model_data['y_scaler'])
            
            # Calculate prediction metrics
            actual_values = sessions[:24]
            mse, mae, r2 = calculate_metrics(actual_values, predictions.flatten())
            
            # Display prediction metrics
            st.markdown("### Prediction Metrics")
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            with pred_col1:
                st.metric("MSE", f"{mse:.4f}")
            with pred_col2:
                st.metric("MAE", f"{mae:.4f}")
            with pred_col3:
                st.metric("R² Score", f"{r2:.4f}")
            
            # Create plots
            fig = go.Figure()
            
            # Actual values
            fig.add_trace(go.Scatter(
                x=list(range(24)),
                y=actual_values,
                name="Actual Traffic",
                line=dict(color="#1f77b4", width=2)
            ))
            
            # Predicted values
            fig.add_trace(go.Scatter(
                x=list(range(24)),
                y=predictions.flatten(),
                name="Predicted Traffic",
                line=dict(color="#ff7f0e", width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="24-Hour Web Traffic Forecast",
                xaxis_title="Hours",
                yaxis_title="Sessions",
                template="plotly_white",
                height=600
            )
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Download predictions
            predictions_df = pd.DataFrame({
                'Hour': range(24),
                'Actual_Traffic': actual_values,
                'Predicted_Traffic': predictions.flatten(),
                'Absolute_Error': abs(actual_values - predictions.flatten())
            })
            
            st.download_button(
                label="Download Predictions",
                data=predictions_df.to_csv(index=False).encode('utf-8'),
                file_name='traffic_predictions.csv',
                mime='text/csv'
            )
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure your CSV file has a 'Sessions' column with hourly web traffic data.")

else:
    st.info("Please upload a CSV file to begin forecasting.")