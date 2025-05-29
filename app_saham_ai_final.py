
import streamlit as st
import sys
import os
import json
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ===== Konfigurasi Awal =====
st.set_page_config(layout="wide", page_title="Analisis Saham - Prophet & LSTM")

# ===== Dependency Fallback =====
try:
    import yfinance as yf
    YFINANCE_ENABLED = True
except ImportError:
    st.sidebar.error("⚠️ yfinance belum terinstal.")
    YFINANCE_ENABLED = False

try:
    from prophet import Prophet
    PROPHET_ENABLED = True
except ImportError:
    st.sidebar.error("⚠️ prophet belum terinstal.")
    PROPHET_ENABLED = False

# ===== Ambil Data Saham =====
def ambil_data_saham(ticker, cache_dir="cache", ttl_jam=1):
    if not YFINANCE_ENABLED:
        return pd.DataFrame(), {}
    os.makedirs(cache_dir, exist_ok=True)
    path_hist = os.path.join(cache_dir, f"{ticker}_hist.csv")
    now = datetime.now()
    def cache_valid(path):
        return os.path.exists(path) and (now - datetime.fromtimestamp(os.path.getmtime(path))) < timedelta(hours=ttl_jam)
    if cache_valid(path_hist):
        try:
            return pd.read_csv(path_hist, index_col=0, parse_dates=True), {}
        except:
            pass
    try:
        saham = yf.Ticker(ticker)
        hist = saham.history(period="1y", interval="1d")
        if not hist.empty:
            hist.to_csv(path_hist)
            return hist, {}
    except:
        pass
    return pd.DataFrame(), {}

# ===== Format =====
def format_rupiah(x):
    try:
        return f"Rp{round(x):,}".replace(",", ".")
    except:
        return "Rp0"

# ===== Prediksi Prophet =====
def prediksi_harga_saham_prophet(ticker, periode_hari=30):
    if not PROPHET_ENABLED:
        st.warning("Modul Prophet tidak tersedia")
        return
    hist, _ = ambil_data_saham(ticker)
    if hist.empty:
        st.warning("Tidak ada data historis untuk prediksi.")
        return
    df = hist[['Close']].reset_index()
    df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=periode_hari)
    forecast = model.predict(future)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Harga Aktual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prediksi Harga'))
    fig.update_layout(title=f"Prediksi Harga Saham {ticker} ({periode_hari} Hari ke Depan)",
                      xaxis_title="Tanggal", yaxis_title="Harga")
    st.plotly_chart(fig, use_container_width=True)
    st.write("### Tabel Prediksi")
    prediksi_tampil = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periode_hari)
    prediksi_tampil.columns = ['Tanggal', 'Prediksi', 'Batas Bawah', 'Batas Atas']
    prediksi_tampil['Prediksi'] = prediksi_tampil['Prediksi'].apply(format_rupiah)
    st.dataframe(prediksi_tampil)

# ===== Prediksi LSTM =====
def lstm_prediksi_harga(ticker, pred_hari=5):
    st.info(f"⏳ Menyiapkan prediksi LSTM untuk {ticker}...")
    data, _ = ambil_data_saham(ticker)
    if data.empty:
        st.error("Data historis tidak tersedia.")
        return
    df = data[["Close"]]
    dataset = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    seq_len = 60
    x, y = [], []
    for i in range(seq_len, len(scaled) - pred_hari):
        x.append(scaled[i - seq_len:i])
        y.append(scaled[i:i + pred_hari, 0])
    x, y = np.array(x), np.array(y)
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(pred_hari))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=10, batch_size=16, verbose=0)
    input_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
    pred_scaled = model.predict(input_seq)
    pred = scaler.inverse_transform(pred_scaled)[0]
    pred_df = pd.DataFrame({
        "Tanggal": pd.date_range(start=df.index[-1] + timedelta(days=1), periods=pred_hari),
        "Prediksi": pred
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df["Close"].values[-100:], name="Harga Historis"))
    fig.add_trace(go.Scatter(x=pred_df["Tanggal"], y=pred_df["Prediksi"], name="Prediksi LSTM"))
    fig.update_layout(title=f"Prediksi Harga Saham {ticker} (LSTM)", xaxis_title="Tanggal", yaxis_title="Harga")
    st.plotly_chart(fig, use_container_width=True)
    st.write("### Tabel Prediksi")
    st.dataframe(pred_df.assign(Prediksi=pred_df["Prediksi"].apply(format_rupiah)))
    perubahan = pred[-1] - df["Close"].values[-1]
    if perubahan > 0.5:
        st.success("🟢 Sinyal: BELI (harga diprediksi naik)")
    elif perubahan < -0.5:
        st.error("🔴 Sinyal: JUAL (harga diprediksi turun)")
    else:
        st.info("⚪ Sinyal: TAHAN (tidak banyak berubah)")

# ===== MAIN APP =====
def main():
    st.title("📊 Analisis Saham + AI Prediksi (Prophet & LSTM)")
    ticker = st.text_input("Masukkan kode saham (misal: UNVR.JK)", value="UNVR.JK")
    st.subheader("🔮 Prediksi Harga - Prophet")
    periode = st.slider("Periode Prediksi Prophet (hari ke depan):", min_value=7, max_value=90, value=30)
    if st.button("Jalankan Prediksi Prophet"):
        prediksi_harga_saham_prophet(ticker, periode)
    st.subheader("📈 Prediksi Harga - LSTM")
    periode_lstm = st.slider("Periode Prediksi LSTM (hari):", 1, 7, 3)
    if st.button("Jalankan Prediksi LSTM"):
        lstm_prediksi_harga(ticker, pred_hari=periode_lstm)

if __name__ == "__main__":
    main()
