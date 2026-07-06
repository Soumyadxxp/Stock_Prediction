import os
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------
# 1. Model Definition
# ---------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # last timestep
        out = self.fc(out)
        return out

# ---------------------------
# 2. Training / Loading (with caching)
# ---------------------------
@st.cache_resource
def load_or_train_model(stock_symbol, train_data):
    """
    Returns a trained model and the scaler fitted on the training data.
    The model is saved as '{stock_symbol}_model.pth' and cached in Streamlit.
    """
    model_path = f"{stock_symbol}_model.pth"
    seq_len = 100

    # Scale the training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1))

    # Create training sequences
    X_train, y_train = [], []
    for i in range(seq_len, len(scaled_train)):
        X_train.append(scaled_train[i-seq_len:i])
        y_train.append(scaled_train[i, 0])
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32).view(-1, 1)

    # If model already exists, load and return
    if os.path.exists(model_path):
        model = StockLSTM()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model, scaler

    # Otherwise train a new model
    model = StockLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 64
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    epochs = 50
    progress_bar = st.progress(0)
    status_text = st.empty()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Update progress
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs} – Loss: {total_loss/len(loader):.6f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    status_text.empty()
    progress_bar.empty()
    st.success("✅ Model trained and saved.")

    return model, scaler

# ---------------------------
# 3. Streamlit UI
# ---------------------------
st.set_page_config(page_title="📈 Stock Predictor", layout="wide")
st.title("📈 Stock Predictor with PyTorch LSTM")
st.markdown("Enter a stock symbol to see its predicted vs actual closing prices.")

# User input
stock = st.text_input("Stock Symbol", value="GOOG").upper()
start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))

if st.button("Run Prediction"):
    with st.spinner("Downloading data..."):
        data = yf.download(stock, start=start_date, end=end_date)

    if data.empty:
        st.error("❌ No data found for that symbol. Please check the ticker and date range.")
        st.stop()

    # Show raw data
    st.subheader("📊 Raw Stock Data")
    st.write(data.tail(10))

    # Split into train/test (80/20)
    train_size = int(len(data) * 0.80)
    train_data = pd.DataFrame(data.Close[:train_size])
    test_data = pd.DataFrame(data.Close[train_size:])

    # Load or train model
    with st.spinner("Preparing model..."):
        model, scaler = load_or_train_model(stock, train_data)

    # Prepare test data (add last 100 days from training for sequence)
    last_100_train = train_data.tail(100)
    test_extended = pd.concat([last_100_train, test_data], ignore_index=True)
    test_scaled = scaler.transform(test_extended.values.reshape(-1, 1))

    # Build test sequences
    seq_len = 100
    X_test, y_test = [], []
    for i in range(seq_len, len(test_scaled)):
        X_test.append(test_scaled[i-seq_len:i])
        y_test.append(test_scaled[i, 0])
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test = np.array(y_test)

    # Predict
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test).numpy().flatten()

    # Inverse scaling
    scale = 1.0 / scaler.scale_[0]
    pred_actual = pred_scaled * scale
    y_actual = y_test * scale

    # Plot
    st.subheader("📉 Prediction vs Actual (Test Set)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pred_actual, 'r', label='Predicted Price', linewidth=2)
    ax.plot(y_actual, 'g', label='Actual Price', linewidth=2)
    ax.set_xlabel("Time (test sequence)")
    ax.set_ylabel("Closing Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Optional: show numeric metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(y_actual, pred_actual)
    mae = mean_absolute_error(y_actual, pred_actual)
    st.metric("Mean Squared Error", f"{mse:.2f}")
    st.metric("Mean Absolute Error", f"{mae:.2f}")

    st.success("🎉 Prediction complete!")