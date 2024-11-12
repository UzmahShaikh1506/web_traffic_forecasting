# Web Traffic Forecasting

This project uses machine learning models to forecast website traffic based on historical data. By analyzing previous traffic patterns, the models predict future trends, enabling proactive management of website resources and improved user experience.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies and Models](#technologies-and-models)
- [Requirements](#requirements)
- [Setup Guide](#setup-guide)
- [Usage](#usage)

---

## Project Overview

The goal of this project is to forecast website traffic using time series data. We approach this problem by creating two models: a Long Short-Term Memory (LSTM) model and a Convolutional Neural Network (CNN) model. Both models are designed to capture the temporal dependencies in web traffic data, though they do so using different architectures suited to specific features in time series data.

### Key Concepts Covered
1. **Data Preprocessing**: Preparing time series data and scaling features for optimal model performance.
2. **Time Series Modeling**: Implementing LSTM and CNN models, both of which have demonstrated effectiveness in predicting sequential data.
3. **Model Evaluation**: Using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score to measure model performance.

## Features

- **LSTM Model**: Captures long-term dependencies in time series data, making it effective for sequential prediction tasks.
- **CNN Model**: Utilizes convolutional layers to identify local patterns in the data, particularly useful for data with repeating patterns over short intervals.
- **Data Scaling**: Standardizes data to ensure stable and efficient model training.
- **Model Evaluation**: Calculates MSE, MAE, and R² score to compare model accuracy and suitability for forecasting.

## Technologies and Models

This project relies on Python libraries for data manipulation, scaling, model building, and evaluation.

1. **Data Preprocessing and Scaling**:
   - `Pandas` and `NumPy` are used for data manipulation.
   - `StandardScaler` from `sklearn.preprocessing` scales both input and output data for more efficient learning.

2. **Models**:
   - **LSTM (Long Short-Term Memory)**: A deep learning model designed to handle long-term dependencies in sequential data. The LSTM model is effective for web traffic data because it learns to retain relevant patterns from previous data points, ideal for predicting traffic peaks and patterns over time.
   - **CNN (Convolutional Neural Network)**: CNN layers extract patterns within a local time frame, useful for detecting short-term fluctuations and repeated patterns in the data. The CNN model captures short-term dependencies and general trends in traffic, complementing the LSTM model.

3. **Model Evaluation**:
   - Metrics include Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score, providing insight into model accuracy.

## Requirements

To run this project, you'll need:
- **Python**: 3.8 or later
- **TensorFlow/Keras**: for building and training neural networks
- **Pandas**: for data manipulation
- **NumPy**: for numerical operations
- **Scikit-learn**: for preprocessing and evaluation metrics

All dependencies are listed in the `requirements.txt` file.

## Setup Guide

Follow these steps to set up the project on your local machine.
### 1. Clone the Repository

```bash
git clone https://github.com/UzmahShaikh1506/web_traffic_forecasting.git
cd web_traffic_forecasting
```
### 2. Set Up a Virtual Environment
It's recommended to use a virtual environment to isolate dependencies.

```bash
python3 -m venv venv
```
### 3. Install Dependencies
Once the virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

## Usage
### Step-by-Step Instructions
**1. Prepare Data:**
- Ensure your data file (webtraffic.csv) is placed in the data/ directory.

Run the Main Script:
```bash
python src/main.py
```
**2. This script performs the following tasks:**

- Loads and preprocesses data.
- Trains both LSTM and CNN models.
- Evaluates each model and saves performance metrics.
- Serializes models and scalers for future use.

**3. Review the Results:**
- The output will display the evaluation metrics for both models, including MSE, MAE, and R² scores.
