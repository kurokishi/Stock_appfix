import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os

# Konfigurasi awal
st.set_page_config(layout="wide", page_title="Analisis Portofolio Saham")

# Fungsi untuk memuat data saham
def load_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker + ".JK")
    hist = stock.history(period=period)
    return stock, hist

# Fungsi untuk menghitung indikator teknikal
def calculate_technical_indicators(df):
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Hitung RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Hitung MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

# Fungsi untuk menampilkan grafik candlestick
def plot_candlestick(df, ticker):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    
    fig.update_layout(
        title=f'Grafik Candlestick {ticker}',
        xaxis_title='Tanggal',
        yaxis_title='Harga (IDR)',
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Fungsi untuk menampilkan indikator teknikal
def plot_technical_indicators(df, ticker):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot harga dan moving averages
    ax1.plot(df.index, df['Close'], label='Harga', color='blue')
    ax1.plot(df.index, df['MA50'], label='MA50', color='orange')
    ax1.plot(df.index, df['MA200'], label='MA200', color='red')
    ax1.set_title(f'Analisis Teknikal {ticker}')
    ax1.legend()
    
    # Plot RSI
    ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--')
    ax2.axhline(30, color='green', linestyle='--')
    ax2.set_ylim(0, 100)
    ax2.legend()
    
    # Plot MACD
    ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax3.plot(df.index, df['Signal'], label='Signal', color='red')
    ax3.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

# Fungsi untuk analisis valuasi (simulasi)
def valuation_analysis(ticker):
    # Dalam implementasi nyata, ini akan mengambil data dari API atau database
    # Di sini kita menggunakan data dummy untuk simulasi
    
    valuations = {
        'PER': np.random.uniform(5, 30),
        'PBV': np.random.uniform(0.5, 5),
        'DCF': np.random.uniform(0.8, 1.5),
        'Dividend Yield': np.random.uniform(0, 0.1)
    }
    
    # Komentar AI sederhana
    if valuations['PER'] < 10 and valuations['PBV'] < 1:
        comment = "Saham undervalued berdasarkan PER dan PBV"
    elif valuations['PER'] > 20 or valuations['PBV'] > 3:
        comment = "Saham overvalued berdasarkan PER dan PBV"
    else:
        comment = "Saham fairly valued"
    
    return valuations, comment

# Fungsi untuk proyeksi dividen
def dividend_projection(ticker, num_lots):
    # Data dummy untuk simulasi
    avg_dividend = np.random.uniform(50, 500)
    growth_rate = np.random.uniform(0, 0.1)
    
    projections = {
        '1 tahun': avg_dividend * num_lots * 100,
        '3 tahun': avg_dividend * (1 + growth_rate)**3 * num_lots * 100,
        '5 tahun': avg_dividend * (1 + growth_rate)**5 * num_lots * 100
    }
    
    return projections

# Fungsi untuk rekomendasi alokasi modal
def allocate_funds(portfolio, new_capital):
    # Algoritma sederhana: alokasikan berdasarkan valuasi terbaik
    recommendations = []
    
    for ticker, data in portfolio.items():
        valuations, _ = valuation_analysis(ticker)
        score = 1/valuations['PER'] + 1/valuations['PBV'] + valuations['Dividend Yield']
        recommendations.append((ticker, score))
    
    # Urutkan berdasarkan score tertinggi
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Alokasikan modal (contoh sederhana: bagi rata ke top 3)
    top_picks = recommendations[:3]
    total_score = sum(score for _, score in top_picks)
    
    allocation = []
    for ticker, score in top_picks:
        percentage = score / total_score
        amount = new_capital * percentage
        allocation.append({
            'Saham': ticker,
            'Alokasi (Rp)': f"{amount:,.0f}",
            'Persentase': f"{percentage*100:.1f}%"
        })
    
    return pd.DataFrame(allocation)

# Fungsi untuk rekomendasi jual/tahan
def sell_hold_recommendation(ticker, avg_price, current_price):
    valuations, comment = valuation_analysis(ticker)
    
    profit_loss = (current_price - avg_price) / avg_price * 100
    
    # Proyeksi harga sederhana
    projection_6m = current_price * (1 + np.random.uniform(-0.2, 0.3))
    projection_12m = current_price * (1 + np.random.uniform(-0.1, 0.5))
    
    # Rekomendasi sederhana
    if profit_loss > 30 and valuations['PER'] > 20:
        action = "Pertimbangkan jual sebagian untuk mengamankan keuntungan"
    elif profit_loss < -20 and valuations['PER'] < 15:
        action = "Bisa ditambahkan posisi (averaging down)"
    else:
        action = "Tahan"
    
    return {
        'Profit/Loss (%)': f"{profit_loss:.1f}%",
        'Proyeksi 6 Bulan (Rp)': f"{projection_6m:,.0f}",
        'Proyeksi 12 Bulan (Rp)': f"{projection_12m:,.0f}",
        'Rekomendasi': action,
        'Komentar Valuasi': comment
    }

# Fungsi untuk simulasi bunga majemuk
def compound_growth_simulation(portfolio, years):
    results = {}
    
    for year in range(1, years + 1):
        total_value = 0
        dividend_income = 0
        
        for ticker, data in portfolio.items():
            # Pertumbuhan harga (random antara -5% sampai 20%)
            price_growth = np.random.uniform(-0.05, 0.2)
            current_value = data['current_price'] * (1 + price_growth) * data['num_lots'] * 100
            total_value += current_value
            
            # Dividen (random antara 1-5% dari harga)
            dividend_yield = np.random.uniform(0.01, 0.05)
            dividend_income += current_value * dividend_yield
        
        # Reinvest dividen
        total_value += dividend_income
        
        results[year] = {
            'Nilai Portofolio (Rp)': f"{total_value:,.0f}",
            'Pendapatan Dividen (Rp)': f"{dividend_income:,.0f}",
            'Pertumbuhan (%)': f"{price_growth*100:.1f}%"
        }
    
    return pd.DataFrame.from_dict(results, orient='index')

# Fungsi untuk menyimpan dan memuat portofolio
def save_portfolio(portfolio):
    with open('portfolio.pkl', 'wb') as f:
        pickle.dump(portfolio, f)

def load_portfolio():
    if os.path.exists('portfolio.pkl'):
        with open('portfolio.pkl', 'rb') as f:
            return pickle.load(f)
    return {}

# Main App
def main():
    st.title("Analisis Portofolio Saham Indonesia")
    
    # Load atau inisialisasi portofolio
    portfolio = load_portfolio()
    
    # Sidebar untuk navigasi
    st.sidebar.title("Menu")
    menu_options = [
        "Analisis Portofolio",
        "Strategi Penambahan Saham",
        "Rekomendasi Jual/Tahan",
        "Edit Portofolio",
        "Simulasi Pertumbuhan"
    ]
    choice = st.sidebar.radio("Pilih Fitur:", menu_options)
    
    # Form edit portofolio
    if choice == "Edit Portofolio":
        st.header("Edit Portofolio Saham")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tambah/Update Saham")
            with st.form("add_stock_form"):
                ticker = st.text_input("Kode Saham (contoh: BBCA)", "").upper()
                num_lots = st.number_input("Jumlah Lot", min_value=1, value=1)
                avg_price = st.number_input("Harga Rata-rata per Saham (IDR)", min_value=1, value=1000)
                
                if st.form_submit_button("Simpan"):
                    try:
                        stock, hist = load_stock_data(ticker)
                        current_price = hist['Close'].iloc[-1]
                        
                        portfolio[ticker] = {
                            'num_lots': num_lots,
                            'avg_price': avg_price,
                            'current_price': current_price,
                            'last_updated': datetime.now().strftime("%Y-%m-%d")
                        }
                        save_portfolio(portfolio)
                        st.success(f"Saham {ticker} berhasil ditambahkan/diperbarui!")
                    except:
                        st.error(f"Gagal mengambil data saham {ticker}. Pastikan kode saham benar.")
        
        with col2:
            st.subheader("Hapus Saham")
            if portfolio:
                delete_ticker = st.selectbox("Pilih saham untuk dihapus", list(portfolio.keys()))
                if st.button("Hapus Saham"):
                    del portfolio[delete_ticker]
                    save_portfolio(portfolio)
                    st.success(f"Saham {delete_ticker} berhasil dihapus!")
            else:
                st.warning("Portofolio kosong")
        
        # Tampilkan portofolio saat ini
        st.subheader("Portofolio Saat Ini")
        if portfolio:
            portfolio_df = pd.DataFrame.from_dict(portfolio, orient='index')
            portfolio_df['Total Investasi'] = portfolio_df['num_lots'] * portfolio_df['avg_price'] * 100
            portfolio_df['Nilai Sekarang'] = portfolio_df['num_lots'] * portfolio_df['current_price'] * 100
            portfolio_df['Profit/Loss'] = (portfolio_df['Nilai Sekarang'] - portfolio_df['Total Investasi']) / portfolio_df['Total Investasi'] * 100
            st.dataframe(portfolio_df.style.format({
                'avg_price': '{:,.0f}',
                'current_price': '{:,.0f}',
                'Total Investasi': '{:,.0f}',
                'Nilai Sekarang': '{:,.0f}',
                'Profit/Loss': '{:.1f}%'
            }))
        else:
            st.info("Portofolio Anda kosong. Tambahkan saham terlebih dahulu.")
    
    # Analisis portofolio
    elif choice == "Analisis Portofolio":
        st.header("Analisis Portofolio Saham")
        
        if not portfolio:
            st.warning("Portofolio Anda kosong. Silakan tambahkan saham terlebih dahulu di menu Edit Portofolio.")
            return
        
        selected_ticker = st.selectbox("Pilih Saham untuk Analisis Mendetail", list(portfolio.keys()))
        
        if selected_ticker:
            data = portfolio[selected_ticker]
            st.subheader(f"Analisis Saham {selected_ticker}")
            
            try:
                stock, hist = load_stock_data(selected_ticker)
                hist = calculate_technical_indicators(hist)
                current_price = hist['Close'].iloc[-1]
                
                # Update current price in portfolio
                portfolio[selected_ticker]['current_price'] = current_price
                save_portfolio(portfolio)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Harga Saat Ini", f"Rp {current_price:,.0f}")
                    st.metric("Harga Rata-rata Anda", f"Rp {data['avg_price']:,.0f}")
                    profit_loss = (current_price - data['avg_price']) / data['avg_price'] * 100
                    st.metric("Profit/Loss", f"{profit_loss:.1f}%", delta_color="inverse")
                    
                    # Valuasi
                    st.subheader("Analisis Valuasi")
                    valuations, comment = valuation_analysis(selected_ticker)
                    for metric, value in valuations.items():
                        st.metric(metric, f"{value:.2f}")
                    st.info(f"Komentar AI: {comment}")
                
                with col2:
                    st.subheader("Proyeksi Dividen")
                    projections = dividend_projection(selected_ticker, data['num_lots'])
                    for period, amount in projections.items():
                        st.metric(f"Dividen {period}", f"Rp {amount:,.0f}")
                    
                    st.subheader("Kinerja Historis")
                    period_option = st.selectbox("Periode", ["1 bulan", "3 bulan", "6 bulan", "1 tahun", "3 tahun"])
                    periods = {
                        "1 bulan": "1mo",
                        "3 bulan": "3mo",
                        "6 bulan": "6mo",
                        "1 tahun": "1y",
                        "3 tahun": "3y"
                    }
                    _, hist_long = load_stock_data(selected_ticker, periods[period_option])
                    price_change = (hist_long['Close'].iloc[-1] - hist_long['Close'].iloc[0]) / hist_long['Close'].iloc[0] * 100
                    st.metric(f"Perubahan Harga ({period_option})", f"{price_change:.1f}%")
                
                # Grafik
                st.subheader("Analisis Teknikal")
                plot_candlestick(hist.tail(100), selected_ticker)
                plot_technical_indicators(hist.tail(100), selected_ticker)
                
            except Exception as e:
                st.error(f"Gagal memuat data saham {selected_ticker}: {str(e)}")
    
    # Strategi penambahan saham
    elif choice == "Strategi Penambahan Saham":
        st.header("Strategi Penambahan Saham Berdasarkan Modal Baru")
        
        if not portfolio:
            st.warning("Portofolio Anda kosong. Silakan tambahkan saham terlebih dahulu di menu Edit Portofolio.")
            return
        
        new_capital = st.number_input("Jumlah Modal Tambahan (IDR)", min_value=100000, value=5000000, step=100000)
        
        if st.button("Buat Rekomendasi Alokasi"):
            st.subheader("Rekomendasi Alokasi Modal")
            allocation = allocate_funds(portfolio, new_capital)
            st.dataframe(allocation)
            
            st.subheader("Penjelasan Rekomendasi")
            st.write("""
            Rekomendasi alokasi modal didasarkan pada:
            - **Valuasi Saham**: Saham dengan PER dan PBV lebih rendah mendapatkan prioritas
            - **Dividend Yield**: Saham dengan yield tinggi mendapatkan poin lebih
            - **Diversifikasi**: Modal dialokasikan ke beberapa saham terbaik untuk mengurangi risiko
            """)
    
    # Rekomendasi jual/tahan
    elif choice == "Rekomendasi Jual/Tahan":
        st.header("Rekomendasi Jual/Tahan Saham")
        
        if not portfolio:
            st.warning("Portofolio Anda kosong. Silakan tambahkan saham terlebih dahulu di menu Edit Portofolio.")
            return
        
        selected_ticker = st.selectbox("Pilih Saham", list(portfolio.keys()))
        
        if selected_ticker:
            data = portfolio[selected_ticker]
            current_price = data['current_price']
            avg_price = data['avg_price']
            
            st.subheader(f"Analisis {selected_ticker}")
            st.write(f"Harga Rata-rata Anda: Rp {avg_price:,.0f}")
            st.write(f"Harga Saat Ini: Rp {current_price:,.0f}")
            
            if st.button("Buat Rekomendasi"):
                recommendation = sell_hold_recommendation(selected_ticker, avg_price, current_price)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Profit/Loss Saat Ini", recommendation['Profit/Loss (%)'])
                    st.metric("Proyeksi Harga 6 Bulan", recommendation['Proyeksi 6 Bulan (Rp)'])
                    st.metric("Proyeksi Harga 12 Bulan", recommendation['Proyeksi 12 Bulan (Rp)'])
                
                with col2:
                    st.metric("Rekomendasi", recommendation['Rekomendasi'])
                    st.info(recommendation['Komentar Valuasi'])
                
                st.subheader("Indikator Risiko/Reward")
                st.write("""
                - **Risiko Tinggi**: Jika saham sudah overvalued dan profit besar (>30%)
                - **Risiko Sedang**: Jika saham fairly valued dengan profit sedang (10-30%)
                - **Risiko Rendah**: Jika saham undervalued dengan kerugian (<-10%)
                """)
    
    # Simulasi pertumbuhan
    elif choice == "Simulasi Pertumbuhan":
        st.header("Simulasi Pertumbuhan Portofolio")
        
        if not portfolio:
            st.warning("Portofolio Anda kosong. Silakan tambahkan saham terlebih dahulu di menu Edit Portofolio.")
            return
        
        years = st.slider("Pilih Jangka Waktu (tahun)", 1, 15, 5)
        
        if st.button("Jalankan Simulasi"):
            st.subheader(f"Proyeksi Pertumbuhan Portofolio {years} Tahun")
            simulation = compound_growth_simulation(portfolio, years)
            st.dataframe(simulation)
            
            # Hitung nilai awal portofolio
            initial_value = sum(data['num_lots'] * data['avg_price'] * 100 for data in portfolio.values())
            
            # Siapkan data untuk grafik
            years_list = list(range(years + 1))
            values = [initial_value]
            
            for year in range(1, years + 1):
                year_data = simulation.loc[year]
                value = float(year_data['Nilai Portofolio (Rp)'].replace(',', ''))
                values.append(value)
            
            # Plot grafik
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(years_list, values, marker='o')
            ax.set_title('Proyeksi Nilai Portofolio')
            ax.set_xlabel('Tahun')
            ax.set_ylabel('Nilai (IDR)')
            ax.grid(True)
            
            # Format y-axis
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            
            st.pyplot(fig)
            
            st.subheader("Asumsi Simulasi")
            st.write("""
            - **Pertumbuhan Harga Saham**: Random antara -5% sampai +20% per tahun
            - **Dividend Yield**: Random antara 1-5% dari nilai portofolio, diinvestasikan kembali
            - **Volatilitas**: Tidak memperhitungkan krisis ekonomi atau kejadian luar biasa
            """)

if __name__ == "__main__":
    main()
