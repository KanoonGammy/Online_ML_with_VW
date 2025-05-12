import pandas as pd
import requests
import math
import streamlit as st
import time
from sklearn.metrics import mean_squared_error
from river import linear_model, preprocessing, compose, metrics
import plotly.graph_objects as go

# Step 1: ดึงข้อมูล BTC จาก Binance (ปรับเป็น 1 นาที)
def get_binance_klines(symbol="BTCUSDT", interval="1m", limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df[["time", "close", "volume"]]

# Step 2: เตรียมฟีเจอร์ + Label
def prepare_data(df):
    df["future_close"] = df["close"].shift(-1)
    df["label"] = (df["future_close"] > df["close"]).astype(int)
    df["label_text"] = df["label"].map({1: "UP", 0: "DOWN"})
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma_diff"] = df["close"] - df["ma5"]
    df = df.dropna()
    return df

# Streamlit Dashboard
st.title("📈 Real-Time BTC Sentiment Prediction (1-min interval) with River")
st.caption("Updated every 60 seconds")

placeholder = st.empty()
countdown_placeholder = st.empty()
update_count = 0

while True:
    with placeholder.container():
        update_count += 1
        df = get_binance_klines()
        df = prepare_data(df)

        model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression()
        )
        metric = metrics.LogLoss()

        y_true = []
        y_pred = []
        label_texts = []
        actual_prices = []
        predicted_prices = []
        mse_list = []

        for i in range(len(df)):
            row = df.iloc[i]
            x = {
                "close": row["close"],
                "volume": row["volume"],
                "ma_diff": row["ma_diff"]
            }
            y = row["label"]
            y_prob = model.predict_proba_one(x).get(True, 0.5)
            model = model.learn_one(x, y)

            y_true.append(y)
            y_pred.append(y_prob)
            label_texts.append(row["label_text"])
            actual_prices.append(row["future_close"])
            predicted_price = row["close"] * (1 + 0.01 * (2 * y_prob - 1))
            predicted_prices.append(predicted_price)

        mse = mean_squared_error(y_true, y_pred)
        mse_list = [mse] * len(df)

        latest = df.iloc[-1]
        latest_x = {
            "close": latest["close"],
            "volume": latest["volume"],
            "ma_diff": latest["ma_diff"]
        }
        latest_pred = model.predict_proba_one(latest_x).get(True, 0.5)

        st.metric("🔮 Prob. price going UP (latest 1-min)", f"{latest_pred:.4f}")
        st.metric("📉 MSE (training set)", f"{mse:.6f}")
        st.metric("🔄 Update count", update_count)

        chart_df = pd.DataFrame({
            "Time": df["time"].iloc[-len(y_pred):],
            "Actual Price": actual_prices,
            "Predicted Price": predicted_prices,
            "Label": label_texts[-len(y_pred):],
            "Prob_UP": y_pred,
            "MSE": mse_list[-len(y_pred):]
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=chart_df["Time"], y=chart_df["Actual Price"], name="Actual Price"))
        fig.add_trace(go.Scatter(x=chart_df["Time"], y=chart_df["Predicted Price"], name="Predicted Price"))
        fig.update_layout(title="BTC Price Prediction", yaxis_range=[103800, 104400], xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(chart_df.tail(10))

    for i in range(60, 0, -1):
        countdown_placeholder.info(f"⏳ Refreshing in {i} seconds...")
        time.sleep(1)
    countdown_placeholder.empty()
    placeholder.empty()
