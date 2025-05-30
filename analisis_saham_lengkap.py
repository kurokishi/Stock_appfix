import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from textblob import TextBlob

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
            df = pd.read_csv(path_hist, index_col=0, parse_dates=True)
            df.index = df.index.tz_localize(None)  # Pastikan tz-naive
            return df, {}
        except:
            pass
    
    try:
        saham = yf.Ticker(ticker)
        hist = saham.history(period="1y", interval="1d")
        if not hist.empty:
            hist.index = hist.index.tz_localize(None)  # Pastikan tz-naive
            hist.to_csv(path_hist)
            return hist, {}
    except:
        pass
    return pd.DataFrame(), {}

#======Multiple input saham=====
def process_multiple_tickers(tickers):
    if len(tickers) == 1:
        # Jika hanya 1 ticker, jalankan seperti biasa
        ticker = tickers[0]
        if app_mode == "Dashboard Utama":
            show_dashboard(ticker)
        elif app_mode == "Analisis Fundamental":
            show_fundamental_analysis(ticker)
        elif app_mode == "Analisis Teknikal":
            show_technical_analysis(ticker)
        elif app_mode == "Prediksi Harga":
            show_price_prediction(ticker)
        elif app_mode == "Simulasi Portofolio":
            portfolio_simulation(ticker)
    else:
        # Jika multiple tickers
        if app_mode == "Perbandingan Saham":
            compare_stocks(tickers)
        elif app_mode == "Dashboard Utama":
            compare_stocks(tickers)  # Default tampilkan perbandingan
        else:
            st.warning("Fitur ini hanya tersedia untuk analisis satu saham")
# ===== Format =====
def format_rupiah(x):
    try:
        return f"Rp{round(x):,}".replace(",", ".")
    except:
        return "Rp0"

# ===== Fundamental Analysis =====
def show_fundamental_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        
        st.subheader("📊 Analisis Fundamental")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Info Perusahaan**")
            info = stock.info
            st.write(f"Nama: {info.get('longName', 'N/A')}")
            st.write(f"Sektor: {info.get('sector', 'N/A')}")
            st.write(f"Industri: {info.get('industry', 'N/A')}")
            st.write(f"Negara: {info.get('country', 'N/A')}")
        
        with col2:
            st.markdown("**Valuasi**")
            st.write(f"P/E: {info.get('trailingPE', 'N/A')}")
            st.write(f"P/B: {info.get('priceToBook', 'N/A')}")
            st.write(f"EPS: {info.get('trailingEps', 'N/A')}")
            st.write(f"Dividen Yield: {info.get('dividendYield', 'N/A')}")
        
        with col3:
            st.markdown("**Kinerja**")
            st.write(f"ROE: {info.get('returnOnEquity', 'N/A')}")
            st.write(f"ROA: {info.get('returnOnAssets', 'N/A')}")
            st.write(f"Profit Margin: {info.get('profitMargins', 'N/A')}")
            st.write(f"Debt/Equity: {info.get('debtToEquity', 'N/A')}")
        
        # Financials chart
        st.markdown("**Laporan Keuangan**")
        financials = stock.financials
        if not financials.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            financials.loc[['Total Revenue', 'Net Income']].T.plot(kind='bar', ax=ax)
            st.pyplot(fig)
    
    except Exception as e:
        st.warning(f"Tidak dapat memuat data fundamental: {str(e)}")

# ===== Technical Indicators =====
def add_technical_indicators(data):
    if data.empty:
        return data
    
    # Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data

def plot_technical_indicators(data, ticker):
    if data.empty:
        return
    
    # Price with Moving Averages
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Harga'))
    fig1.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20'))
    fig1.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50'))
    fig1.update_layout(title=f"Moving Averages - {ticker}", xaxis_title="Tanggal", yaxis_title="Harga")
    st.plotly_chart(fig1, use_container_width=True)
    
    # RSI
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
    fig2.add_hline(y=30, line_dash="dash", line_color="green")
    fig2.add_hline(y=70, line_dash="dash", line_color="red")
    fig2.update_layout(title=f"RSI (14 hari) - {ticker}", xaxis_title="Tanggal", yaxis_title="RSI")
    st.plotly_chart(fig2, use_container_width=True)
    
    # MACD
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD'))
    fig3.add_trace(go.Scatter(x=data.index, y=data['Signal'], name='Signal'))
    fig3.update_layout(title=f"MACD - {ticker}", xaxis_title="Tanggal", yaxis_title="Nilai")
    st.plotly_chart(fig3, use_container_width=True)

# ===== News Sentiment Analysis =====
def get_news_sentiment(ticker):
    try:
        st.subheader("📰 Analisis Sentimen Berita")
        
        # Get company name for better search
        stock = yf.Ticker(ticker)
        company_name = stock.info.get('longName', ticker.split('.')[0])
        
        # Simulate news fetching (in a real app, you'd use a news API)
        st.info(f"Berita terbaru untuk {company_name}")
        
        # Example news (in a real app, fetch actual news)
        example_news = [
            f"{company_name} melaporkan peningkatan pendapatan kuartalan",
            f"Analis merekomendasikan beli saham {company_name}",
            f"{company_name} menghadapi tantangan regulasi baru",
            f"{company_name} mengumumkan dividen yang lebih tinggi"
        ]
        
        sentiments = []
        for news in example_news:
            blob = TextBlob(news)
            sentiment = blob.sentiment.polarity
            sentiments.append(sentiment)
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"• {news}")
            with col2:
                if sentiment > 0.2:
                    st.success("Positif")
                elif sentiment < -0.2:
                    st.error("Negatif")
                else:
                    st.info("Netral")
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        st.metric("Sentimen Rata-rata", 
                 "Positif" if avg_sentiment > 0.1 else "Negatif" if avg_sentiment < -0.1 else "Netral",
                 f"{avg_sentiment:.2f}")
    
    except Exception as e:
        st.warning(f"Tidak dapat memuat analisis sentimen: {str(e)}")

# ===== Portfolio Simulation =====
def portfolio_simulation(ticker):
    st.subheader("💰 Simulasi Portofolio")
    
    data, _ = ambil_data_saham(ticker)
    if data.empty:
        st.warning("Data tidak tersedia untuk simulasi")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        initial_investment = st.number_input("Jumlah Investasi Awal (Rp)", min_value=100000, value=10000000, step=100000)
    with col2:
        investment_date = st.date_input("Tanggal Investasi", 
                                      value=datetime.now() - timedelta(days=180),
                                      min_value=data.index[0].date(),
                                      max_value=data.index[-1].date())
    
    if st.button("Hitung Kinerja"):
        # Konversi ke datetime tanpa timezone
        investment_date = pd.to_datetime(investment_date).tz_localize(None)
        
        if investment_date < data.index[0] or investment_date > data.index[-1]:
            st.error("Tanggal investasi tidak valid")
            return
        
        # Find the closest trading day
        mask = data.index >= investment_date
        if not any(mask):
            st.error("Tidak ada data untuk tanggal tersebut")
            return
        
        start_price = data.loc[mask].iloc[0]['Close']
        current_price = data['Close'].iloc[-1]
        
        shares = initial_investment / start_price
        current_value = shares * current_price
        profit = current_value - initial_investment
        profit_pct = (profit / initial_investment) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nilai Awal", format_rupiah(initial_investment))
        with col2:
            st.metric("Nilai Sekarang", format_rupiah(current_value), 
                      f"{profit_pct:.2f}%")
        with col3:
            st.metric("Keuntungan/Rugi", format_rupiah(profit))
        
        # Plot investment performance
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'] / start_price * initial_investment,
            name='Nilai Portofolio'
        ))
        fig.add_vline(x=investment_date, line_dash="dash", line_color="green")
        fig.update_layout(
            title="Kinerja Portofolio",
            xaxis_title="Tanggal",
            yaxis_title="Nilai (Rp)"
        )
        st.plotly_chart(fig, use_container_width=True)

# ===== Stock Comparison =====
def compare_stocks(tickers):
    st.subheader("🆚 Perbandingan Saham")
    
    if len(tickers) < 2:
        st.warning("Masukkan minimal 2 kode saham untuk dibandingkan")
        return
    
    data = {}
    for ticker in tickers:
        df, _ = ambil_data_saham(ticker)
        if not df.empty:
            data[ticker] = df['Close']
        else:
            st.warning(f"Data untuk {ticker} tidak tersedia")
    
    if len(data) < 2:
        st.error("Tidak cukup data saham yang valid untuk perbandingan")
        return
    
    # Normalisasi harga untuk perbandingan
    comparison_df = pd.DataFrame(data)
    comparison_df = comparison_df.dropna()
    
    if len(comparison_df) == 0:
        st.error("Tidak ada periode yang sama untuk dibandingkan")
        return
    
    comparison_df = comparison_df / comparison_df.iloc[0] * 100
    
    fig = go.Figure()
    for ticker in comparison_df.columns:
        fig.add_trace(go.Scatter(
            x=comparison_df.index,
            y=comparison_df[ticker],
            name=ticker,
            mode='lines'
        ))
    
    fig.update_layout(
        title="Perbandingan Kinerja Saham (Normalisasi 100 pada awal periode)",
        xaxis_title="Tanggal",
        yaxis_title="Kinerja (%)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tampilkan tabel performa
    st.subheader("Performa Relatif")
    start_date = comparison_df.index[0].strftime('%Y-%m-%d')
    end_date = comparison_df.index[-1].strftime('%Y-%m-%d')
    
    performance = {
        'Saham': [],
        'Perubahan (%)': [],
        'Volatilitas (Std Dev)': []
    }
    
    for ticker in comparison_df.columns:
        change_pct = (comparison_df[ticker].iloc[-1] - 100) / 100 * 100
        volatility = comparison_df[ticker].pct_change().std() * np.sqrt(252) * 100  # Annualized
        performance['Saham'].append(ticker)
        performance['Perubahan (%)'].append(f"{change_pct:.2f}%")
        performance['Volatilitas (Std Dev)'].append(f"{volatility:.2f}%")
    
    st.table(pd.DataFrame(performance))
    
    st.caption(f"Periode: {start_date} hingga {end_date}")
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
#====tambahan fungsi====

def show_dashboard(ticker):
    """Fungsi baru untuk tampilan dashboard individual"""
    st.subheader("📈 Grafik Harga Saham")
    data, _ = ambil_data_saham(ticker)
    if not data.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Harga Penutupan'))
        fig.update_layout(title=f"Harga Saham {ticker}", xaxis_title="Tanggal", yaxis_title="Harga (Rp)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Harga Terakhir", format_rupiah(data['Close'].iloc[-1]))
        with col2:
            change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            pct_change = (change / data['Close'].iloc[-2]) * 100
            st.metric("Perubahan Hari Ini", format_rupiah(change), f"{pct_change:.2f}%")
        with col3:
            st.metric("Volume Hari Ini", f"{data['Volume'].iloc[-1]:,}".replace(",", "."))
        
        show_fundamental_analysis(ticker)
        get_news_sentiment(ticker)
    else:
        st.warning("Data saham tidak tersedia")

def show_technical_analysis(ticker):
    """Fungsi baru untuk analisis teknikal individual"""
    st.subheader("📊 Analisis Teknikal")
    data, _ = ambil_data_saham(ticker)
    if not data.empty:
        data = add_technical_indicators(data)
        plot_technical_indicators(data, ticker)
    else:
        st.warning("Data saham tidak tersedia")

def show_price_prediction(ticker):
    """Fungsi baru untuk prediksi harga individual"""
    st.subheader("🔮 Prediksi Harga Saham")
    
    tab1, tab2 = st.tabs(["Prophet", "ARIMA"])
    
    with tab1:
        periode = st.slider("Periode Prediksi Prophet (hari ke depan):", min_value=7, max_value=90, value=30)
        if st.button("Jalankan Prediksi Prophet"):
            prediksi_harga_saham_prophet(ticker, periode)
    
    with tab2:
        periode_arima = st.slider("Periode Prediksi ARIMA (hari):", 1, 30, 7)
        if st.button("Jalankan Prediksi ARIMA"):
            arima_prediksi_harga(ticker, pred_hari=periode_arima)
# ===== MAIN APP =====
def main():
    st.title("📊 Analisis Saham Lengkap + AI Prediksi")
    
    # Sidebar untuk navigasi
    st.sidebar.title("Menu")
    app_mode = st.sidebar.radio("Pilih Analisis", 
                               ["Dashboard Utama", "Analisis Fundamental", 
                                "Analisis Teknikal", "Prediksi Harga", 
                                "Simulasi Portofolio", "Perbandingan Saham"])
    
    # [POIN 4] Input ticker multiple - GANTI bagian input ticker lama
    tickers_input = st.sidebar.text_input(
        "Masukkan kode saham (pisahkan dengan koma)", 
        value="UNVR.JK, BBCA.JK, TLKM.JK"
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    if not tickers:
        st.warning("Silakan masukkan minimal satu kode saham")
        return
    
    # [POIN 5] Auto-fallback logic - TAMBAHKAN blok ini
    if app_mode == "Perbandingan Saham":
        compare_stocks(tickers)
    elif len(tickers) > 1:  # Jika user input multiple ticker di mode non-comparison
        st.warning(f"Mode '{app_mode}' hanya tersedia untuk analisis satu saham")
        st.info("Sedang menampilkan mode Perbandingan Saham sebagai gantinya")
        compare_stocks(tickers)
    else:  # Jika hanya 1 ticker
        ticker = tickers[0]
        if app_mode == "Dashboard Utama":
            show_dashboard(ticker)
        elif app_mode == "Analisis Fundamental":
            show_fundamental_analysis(ticker)
        elif app_mode == "Analisis Teknikal":
            show_technical_analysis(ticker)
        elif app_mode == "Prediksi Harga":
            show_price_prediction(ticker)
        elif app_mode == "Simulasi Portofolio":
            portfolio_simulation(ticker)
