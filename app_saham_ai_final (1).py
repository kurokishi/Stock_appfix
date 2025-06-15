import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize
import json

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

# ===== CRUD Operations for Custom Stock Data =====
def manage_stock_data():
    data_file = "custom_stock_data.json"
    if not os.path.exists(data_file):
        with open(data_file, 'w') as f:
            json.dump({}, f)
    
    with open(data_file, 'r') as f:
        stock_data = json.load(f)
    
    st.subheader("📋 Manajemen Data Saham")
    with st.form("stock_form"):
        ticker = st.text_input("Kode Saham")
        date = st.date_input("Tanggal")
        price = st.number_input("Harga Penutupan", min_value=0.0)
        action = st.selectbox("Aksi", ["Tambah", "Update", "Hapus"])
        submit = st.form_submit_button("Proses")
        
        if submit and ticker:
            date_str = date.strftime('%Y-%m-%d')
            if action == "Tambah" or action == "Update":
                if ticker not in stock_data:
                    stock_data[ticker] = {}
                stock_data[ticker][date_str] = price
            elif action == "Hapus" and ticker in stock_data:
                if date_str in stock_data[ticker]:
                    del stock_data[ticker][date_str]
                if not stock_data[ticker]:
                    del stock_data[ticker]
            
            with open(data_file, 'w') as f:
                json.dump(stock_data, f)
            st.success(f"Data untuk {ticker} berhasil di{action.lower()}!")
    
    if stock_data:
        st.write("Data Saham Tersimpan:")
        for ticker, prices in stock_data.items():
            st.write(f"{ticker}: {len(prices)} data poin")

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
        info = saham.info
        if not hist.empty:
            hist.to_csv(path_hist)
            return hist, info
    except:
        pass
    
    # Fallback to custom data if available
    data_file = "custom_stock_data.json"
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            custom_data = json.load(f)
        if ticker in custom_data:
            df = pd.DataFrame.from_dict(custom_data[ticker], orient='index', columns=['Close'])
            df.index = pd.to_datetime(df.index)
            return df, {}
    
    return pd.DataFrame(), {}

# ===== Format Rupiah =====
def format_rupiah(x):
    try:
        return f"Rp{round(x):,}".replace(",", ".")
    except:
        return "Rp0"

# ===== Analisis Risiko =====
def analisis_risiko(ticker):
    data, _ = ambil_data_saham(ticker)
    if data.empty:
        st.error("Data tidak tersedia untuk analisis risiko.")
        return
    
    returns = data['Close'].pct_change().dropna()
    
    # Hitung volatilitas
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    # Hitung Value at Risk (VaR) 95%
    var_95 = np.percentile(returns, 5) * np.sqrt(252)  # Annualized VaR
    
    st.subheader("📉 Analisis Risiko")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Volatilitas (Annual)", f"{volatility*100:.2f}%")
    with col2:
        st.metric("Value at Risk (95%)", f"{var_95*100:.2f}%")
    
    # Visualisasi distribusi return
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, name="Distribusi Return"))
    fig.update_layout(title=f"Distribusi Return Harian {ticker}",
                     xaxis_title="Return Harian", yaxis_title="Frekuensi")
    st.plotly_chart(fig, use_container_width=True)

# ===== Benchmarking =====
def benchmarking(ticker, benchmark_ticker="^JKSE"):
    data_stock, _ = ambil_data_saham(ticker)
    data_bench, _ = ambil_data_saham(benchmark_ticker)
    
    if data_stock.empty or data_bench.empty:
        st.error("Data tidak tersedia untuk benchmarking.")
        return
    
    returns_stock = data_stock['Close'].pct_change().dropna()
    returns_bench = data_bench['Close'].pct_change().dropna()
    
    # Align dates
    common_dates = returns_stock.index.intersection(returns_bench.index)
    returns_stock = returns_stock.loc[common_dates]
    returns_bench = returns_bench.loc[common_dates]
    
    # Hitung korelasi dan beta
    correlation = returns_stock.corr(returns_bench)
    beta = np.cov(returns_stock, returns_bench)[0,1] / np.var(returns_bench)
    
    st.subheader("📊 Benchmarking vs Indeks")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Korelasi dengan Indeks", f"{correlation:.2f}")
    with col2:
        st.metric("Beta", f"{beta:.2f}")
    
    # Visualisasi perbandingan
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns_stock.index, y=returns_stock.cumsum(), name=ticker))
    fig.add_trace(go.Scatter(x=returns_bench.index, y=returns_bench.cumsum(), name=benchmark_ticker))
    fig.update_layout(title="Perbandingan Return Kumulatif",
                     xaxis_title="Tanggal", yaxis_title="Return Kumulatif")
    st.plotly_chart(fig, use_container_width=True)

# ===== Dividend Tracking =====
def dividend_tracking(ticker):
    if not YFINANCE_ENABLED:
        st.error("yfinance diperlukan untuk dividend tracking.")
        return
    
    try:
        saham = yf.Ticker(ticker)
        dividends = saham.dividends
        if dividends.empty:
            st.warning("Tidak ada data dividen untuk saham ini.")
            return
        
        # Hitung dividend yield
        current_price = saham.history(period="1d")['Close'][-1]
        annual_dividend = dividends[-4:].sum() if len(dividends) >= 4 else dividends.sum()
        dividend_yield = (annual_dividend / current_price) * 100
        
        st.subheader("💰 Pelacakan Dividen")
        st.metric("Dividend Yield (TTM)", f"{dividend_yield:.2f}%")
        
        # Visualisasi dividen
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dividends.index, y=dividends.values, name="Dividen"))
        fig.update_layout(title=f"Riwayat Dividen {ticker}",
                         xaxis_title="Tanggal", yaxis_title="Jumlah Dividen")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### Detail Dividen")
        st.dataframe(dividends.tail())
    except:
        st.error("Gagal mengambil data dividen.")

# ===== Optimasi Alokasi Portofolio =====
def portfolio_optimization(tickers):
    st.subheader("📈 Optimasi Portofolio")
    if len(tickers) < 2:
        st.warning("Masukkan minimal 2 saham untuk optimasi portofolio.")
        return
    
    returns_data = []
    for ticker in tickers:
        data, _ = ambil_data_saham(ticker)
        if not data.empty:
            returns = data['Close'].pct_change().dropna()
            returns_data.append(returns)
    
    if len(returns_data) < 2:
        st.error("Data tidak cukup untuk optimasi portofolio.")
        return
    
    # Align dates
    returns_df = pd.concat(returns_data, axis=1).dropna()
    returns_df.columns = tickers
    
    # Hitung return dan kovarians
    returns_mean = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    
    def portfolio_performance(weights, returns, cov_matrix):
        port_return = np.sum(returns * weights) * 252
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        return port_return, port_std
    
    def minimize_volatility(weights, returns, cov_matrix):
        return portfolio_performance(weights, returns, cov_matrix)[1]
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    init_guess = [1./len(tickers)] * len(tickers)
    
    opt_results = minimize(minimize_volatility, init_guess,
                         args=(returns_mean, cov_matrix),
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = opt_results.x
    opt_return, opt_std = portfolio_performance(optimal_weights, returns_mean, cov_matrix)
    
    st.write("### Alokasi Portofolio Optimal")
    for ticker, weight in zip(tickers, optimal_weights):
        st.write(f"{ticker}: {weight*100:.2f}%")
    st.metric("Expected Annual Return", f"{opt_return*100:.2f}%")
    st.metric("Portfolio Volatility", f"{opt_std*100:.2f}%")
    
    # Visualisasi alokasi
    fig = go.Figure()
    fig.add_trace(go.Pie(labels=tickers, values=optimal_weights))
    fig.update_layout(title="Alokasi Portofolio Optimal")
    st.plotly_chart(fig, use_container_width=True)

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

# ===== Prediksi ARIMA =====
def arima_prediksi_harga(ticker, pred_hari=30):
    st.info(f"⏳ Menyiapkan prediksi ARIMA untuk {ticker}...")
    data, _ = ambil_data_saham(ticker)
    if data.empty:
        st.error("Data historis tidak tersedia.")
        return
    
    df = data[["Close"]]
    
    st.write("🔍 Mencari parameter ARIMA terbaik...")
    best_aic = float("inf")
    best_order = None
    
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
    
    model = ARIMA(df, order=best_order)
    model_fit = model.fit()
    
    forecast = model_fit.get_forecast(steps=pred_hari)
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    pred_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=pred_hari)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df["Close"].values[-100:], name="Harga Historis", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=pred_dates, y=pred_mean, name="Prediksi", line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=pred_dates.tolist() + pred_dates.tolist()[::-1],
                           y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1].tolist()[::-1],
                           fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'),
                           name="Interval Kepercayaan 95%"))
    
    fig.update_layout(title=f"Prediksi Harga Saham {ticker} (ARIMA{best_order})",
                     xaxis_title="Tanggal", yaxis_title="Harga", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
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
    
    perubahan = pred_mean.iloc[-1] - df["Close"].iloc[-1]
    perubahan_persen = (perubahan / df["Close"].iloc[-1]) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediksi Akhir", f"Rp{pred_mean.iloc[-1]:,.2f}".replace(",", "."), f"{perubahan_persen:+.2f}%")
    
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
    
    ticker = st.text_input("Masukkan kode saham (misal: UNVR.JK)", value="UNVR.JK")
    
    # CRUD Operations
    manage_stock_data()
    
    # Risk Analysis
    st.subheader("🔍 Analisis Risiko")
    if st.button("Lakukan Analisis Risiko"):
        analisis_risiko(ticker)
    
    # Benchmarking
    st.subheader("📊 Benchmarking")
    benchmark_ticker = st.text_input("Kode Indeks Benchmark", value="^JKSE")
    if st.button("Lakukan Benchmarking"):
        benchmarking(ticker, benchmark_ticker)
    
    # Dividend Tracking
    st.subheader("💰 Pelacakan Dividen")
    if st.button("Tampilkan Data Dividen"):
        dividend_tracking(ticker)
    
    # Portfolio Optimization
    st.subheader("📈 Optimasi Portofolio")
    portfolio_tickers = st.text_area("Masukkan kode saham (pisahkan dengan koma)", value="UNVR.JK,BBCA.JK,TLKM.JK")
    if st.button("Optimasi Portofolio"):
        tickers = [t.strip() for t in portfolio_tickers.split(",")]
        portfolio_optimization(tickers)
    
    # Prophet Prediction
    st.subheader("🔮 Prediksi Harga - Prophet")
    periode = st.slider("Periode Prediksi Prophet (hari ke depan):", min_value=7, max_value=90, value=30)
    if st.button("Jalankan Prediksi Prophet"):
        prediksi_harga_saham_prophet(ticker, periode)
    
    # ARIMA Prediction
    st.subheader("📈 Prediksi Harga - ARIMA")
    periode_arima = st.slider("Periode Prediksi ARIMA (hari):", 1, 30, 7)
    if st.button("Jalankan Prediksi ARIMA"):
        arima_prediksi_harga(ticker, pred_hari=periode_arima)

if __name__ == "__main__":
    main()