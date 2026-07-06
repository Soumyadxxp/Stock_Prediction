# Stock Price Predictor using LSTM (PyTorch + Streamlit)

This project is a machine learning web application that predicts stock closing prices using a deep learning LSTM (Long Short-Term Memory) model. The system uses historical stock data from Yahoo Finance and provides an interactive web interface built with Streamlit.

---

## Features

- Stock price prediction using LSTM neural network
- Historical data fetched using Yahoo Finance API
- Automatic model training and saving
- Reusable trained models (no retraining required once saved)
- Prediction vs actual price visualization
- Model evaluation using MSE and MAE
- Interactive Streamlit web interface

---

## Tech Stack

- Python
- PyTorch (Deep Learning Framework)
- Streamlit (Web Application Framework)
- Yahoo Finance (yfinance)
- Pandas and NumPy (Data Processing)
- Scikit-learn (Preprocessing and Evaluation)
- Matplotlib (Visualization)

---

## How It Works

1. User enters a stock symbol (example: AAPL, TSLA, GOOG)
2. Historical stock data is downloaded using Yahoo Finance
3. Data is preprocessed and scaled using MinMaxScaler
4. LSTM model is trained using sequences of past 100 closing prices
5. Model predicts stock prices on test data
6. Results are displayed using graphs and metrics

---

## Model Architecture

- Input sequence length: 100 timesteps
- LSTM layers: 2
- Hidden units: 50
- Fully connected output layer: 1 neuron
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam optimizer (learning rate 0.001)

---

## Installation

### Step 1: Clone the repository
git clone https://github.com/your-username/stock-predictor.git
cd stock-predictor

---

### Step 2: Install dependencies
pip install -r requirements.txt

---

## Requirements

streamlit
torch
numpy
pandas
yfinance
scikit-learn
matplotlib

---

## Run the Application

streamlit run app.py

---

## Usage

1. Enter a stock ticker symbol (example: AAPL)
2. Select start and end dates
3. Click on Run Prediction
4. View:
   - Raw stock data
   - Prediction vs actual graph
   - MSE and MAE values

---

## Model Saving

Trained models are saved automatically as:

{stock_symbol}_model.pth

This allows reuse of models without retraining.

---

## Output

The application displays:

- Actual stock closing prices
- Predicted stock prices
- Error metrics (MSE, MAE)
- Interactive visualization graph

---

## Limitations

- Works best for historical pattern-based data
- Not suitable for real-time trading decisions
- Accuracy depends on stock volatility and dataset size
- LSTM cannot capture sudden market crashes or news impacts

---

## Future Improvements

- Add technical indicators (RSI, MACD, Moving Averages)
- Use Transformer-based models for better accuracy
- Add multi-feature prediction (volume, open, high, low)
- Improve UI with dashboards and charts
- Add real-time prediction updates

---

## Author

Soumyadeep Basu

This project is created for educational purposes to demonstrate time series forecasting using deep learning and Streamlit.
