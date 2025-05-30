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
def arima_prediksi_harga(ticker, pred_hari=30):  # Maksimal prediksi diubah menjadi 30 hari
    st.info(f"⏳ Menyiapkan prediksi ARIMA untuk {ticker}...")
    data, _ = ambil_data_saham(ticker)
    if data.empty:
        st.error("Data historis tidak tersedia.")
        return
    
    df = data[["Close"]]
    
    # 1. Optimasi parameter ARIMA secara otomatis
    st.write("🔍 Mencari parameter ARIMA terbaik...")
    best_aic = float("inf")
    best_order = None
    
    # Coba kombinasi parameter yang masuk akal
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    model = ARIMA(df, order=(p,d,q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p,d,q)
                except:
                    continue
    
    if best_order is None:
        st.error("Gagal menemukan model ARIMA yang cocok")
        return
    
    st.success(f"✅ Model terbaik: ARIMA{best_order} dengan AIC: {best_aic:.2f}")
    
    # 2. Training model dengan parameter terbaik
    model = ARIMA(df, order=best_order)
    model_fit = model.fit()
    
    # 3. Prediksi dengan interval kepercayaan
    forecast = model_fit.get_forecast(steps=pred_hari)
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    # 4. Persiapan data untuk visualisasi
    pred_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=pred_hari)
    
    # 5. Visualisasi yang lebih informatif
    fig = go.Figure()
    
    # Plot data historis
    fig.add_trace(go.Scatter(
        x=df.index[-100:], 
        y=df["Close"].values[-100:],
        name="Harga Historis",
        line=dict(color='blue')
    ))
    
    # Plot prediksi
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=pred_mean,
        name="Prediksi",
        line=dict(color='green', dash='dash')
    ))
    
    # Plot interval kepercayaan
    fig.add_trace(go.Scatter(
        x=pred_dates.tolist() + pred_dates.tolist()[::-1],
        y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name="Interval Kepercayaan 95%"
    ))
    
    fig.update_layout(
        title=f"Prediksi Harga Saham {ticker} (ARIMA{best_order})",
        xaxis_title="Tanggal",
        yaxis_title="Harga",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 6. Tampilkan tabel prediksi
    pred_df = pd.DataFrame({
        "Tanggal": pred_dates,
        "Prediksi": pred_mean,
        "Bawah (95%)": conf_int.iloc[:, 0],
        "Atas (95%)": conf_int.iloc[:, 1]
    })
    
    st.write("### Detail Prediksi")
    st.dataframe(pred_df.style.format({
        "Prediksi": "{:.2f}",
        "Bawah (95%)": "{:.2f}",
        "Atas (95%)": "{:.2f}"
    }))
    
    # 7. Analisis perubahan harga
    perubahan = pred_mean.iloc[-1] - df["Close"].iloc[-1]
    perubahan_persen = (perubahan / df["Close"].iloc[-1]) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Prediksi Akhir",
            f"Rp{pred_mean.iloc[-1]:,.2f}".replace(",", "."),
            f"{perubahan_persen:+.2f}%"
        )
    
    with col2:
        if perubahan > 0.5:
            st.success("🟢 Sinyal: BELI (harga diprediksi naik)")
        elif perubahan < -0.5:
            st.error("🔴 Sinyal: JUAL (harga diprediksi turun)")
        else:
            st.info("⚪ Sinyal: TAHAN (tidak banyak berubah)")

# ===== MAIN APP =====
def main():
    st.title("📊 Analisis Saham + AI Prediksi (Prophet & ARIMA)")
    
    # Input ticker harus didefinisikan di sini sebelum digunakan
    ticker = st.text_input("Masukkan kode saham (misal: UNVR.JK)", value="UNVR.JK")
    
    st.subheader("🔮 Prediksi Harga - Prophet")
    periode = st.slider("Periode Prediksi Prophet (hari ke depan):", min_value=7, max_value=90, value=30)
    if st.button("Jalankan Prediksi Prophet"):
        prediksi_harga_saham_prophet(ticker, periode)
    
    st.subheader("📈 Prediksi Harga - ARIMA")
    periode_arima = st.slider("Periode Prediksi ARIMA (hari):", 1, 30, 7)
    if st.button("Jalankan Prediksi ARIMA"):
        arima_prediksi_harga(ticker, pred_hari=periode_arima)  # Pastikan ticker dikirim sebagai parameter
    
if __name__ == "__main__":
    main()
