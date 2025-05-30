import streamlit as st
import os  # Tambahkan ini di bagian import
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
# ===== Konfigurasi Awal =====
st.set_page_config(layout="wide", page_title="Analisis Saham - Prophet & ARIMA")

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

# ===== Ambil Data Saham (Sama seperti sebelumnya) =====
def ambil_data_saham(ticker, cache_dir="cache", ttl_jam=1):
    if not YFINANCE_ENABLED:
        return pd.DataFrame(), {}
    os.makedirs(cache_dir, exist_ok=True)  # Ini membutuhkan modul os
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

# ===== Prediksi Prophet (Tetap sama) =====
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

# ===== Prediksi ARIMA (Pengganti LSTM) =====
def arima_prediksi_harga(ticker, pred_hari=5):
    st.info(f"⏳ Menyiapkan prediksi ARIMA untuk {ticker}...")
    data, _ = ambil_data_saham(ticker)
    if data.empty:
        st.error("Data historis tidak tersedia.")
        return
    
    df = data[["Close"]]
    
    # Normalisasi data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Model ARIMA
    model = ARIMA(scaled_data, order=(5,1,0))  # Parameter sederhana
    model_fit = model.fit()
    
    # Prediksi
    forecast = model_fit.forecast(steps=pred_hari)
    forecast = scaler.inverse_transform(forecast.reshape(-1, 1))
    
    pred_df = pd.DataFrame({
        "Tanggal": pd.date_range(start=df.index[-1] + timedelta(days=1), periods=pred_hari),
        "Prediksi": forecast.flatten()
    })
    
    # Visualisasi
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df["Close"].values[-100:], name="Harga Historis"))
    fig.add_trace(go.Scatter(x=pred_df["Tanggal"], y=pred_df["Prediksi"], name="Prediksi ARIMA"))
    fig.update_layout(title=f"Prediksi Harga Saham {ticker} (ARIMA)", 
                     xaxis_title="Tanggal", yaxis_title="Harga")
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("### Tabel Prediksi")
    st.dataframe(pred_df.assign(Prediksi=pred_df["Prediksi"].apply(format_rupiah)))
    
    # Sinyal sederhana
    perubahan = pred_df["Prediksi"].iloc[-1] - df["Close"].values[-1]
    if perubahan > 0.5:
        st.success("🟢 Sinyal: BELI (harga diprediksi naik)")
    elif perubahan < -0.5:
        st.error("🔴 Sinyal: JUAL (harga diprediksi turun)")
    else:
        st.info("⚪ Sinyal: TAHAN (tidak banyak berubah)")

# ===== MAIN APP =====
def main():
    st.title("📊 Analisis Saham + AI Prediksi (Prophet & ARIMA)")
    ticker = st.text_input("Masukkan kode saham (misal: UNVR.JK)", value="UNVR.JK")
    
    st.subheader("🔮 Prediksi Harga - Prophet")
    periode = st.slider("Periode Prediksi Prophet (hari ke depan):", min_value=7, max_value=90, value=30)
    if st.button("Jalankan Prediksi Prophet"):
        prediksi_harga_saham_prophet(ticker, periode)
    
    st.subheader("📈 Prediksi Harga - ARIMA")
    periode_arima = st.slider("Periode Prediksi ARIMA (hari):", 1, 7, 3)
    if st.button("Jalankan Prediksi ARIMA"):
        arima_prediksi_harga(ticker, pred_hari=periode_arima)

if __name__ == "__main__":
    main()
