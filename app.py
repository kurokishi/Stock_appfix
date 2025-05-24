import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from ta import momentum, trend

# Dummy AI comment function
def ai_comment(valuation):
    if valuation < 0.9:
        return "Undervalued"
    elif 0.9 <= valuation <= 1.1:
        return "Fairly Valued"
    else:
        return "Overvalued"

# Load and update portfolio
def load_portfolio():
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame({
            'Kode': ['ADRO', 'ANTM', 'BFIN', 'BJBR', 'BSSR', 'PTBA', 'UNVR', 'WIIM', 'PGAS'],
            'Lot': [17, 15, 30, 23, 10, 4, 60, 35, 10],
            'Harga_Beli': [2605, 1423, 1080, 1145, 4500, 2400, 1860, 871, 1600]
        })
    return st.session_state.portfolio

def update_portfolio(portfolio):
    st.session_state.portfolio = portfolio

# Fundamental & technical analysis
def analyze_stock(stock_code):
    try:
        ticker = yf.Ticker(stock_code + ".JK")
        hist = ticker.history(period="1y")
        info = ticker.info

        # Valuation ratios (dummy PBV/PER if missing)
        pbv = info.get("priceToBook", np.nan)
        per = info.get("trailingPE", np.nan)
        div_yield = info.get("dividendYield", 0) * 100
        fair_price = info.get("targetMeanPrice", np.nan)
        current_price = hist["Close"].iloc[-1]

        valuation_ratio = current_price / fair_price if fair_price else np.nan
        comment = ai_comment(valuation_ratio)

        # Technical indicators
        rsi = momentum.RSIIndicator(hist['Close']).rsi().iloc[-1]
        macd = momentum.MACD(hist['Close']).macd_diff().iloc[-1]
        ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]

        return {
            "Kode": stock_code,
            "Harga_Terakhir": current_price,
            "PBV": pbv,
            "PER": per,
            "Dividend_Yield": div_yield,
            "Harga_Wajar": fair_price,
            "Komentar_AI": comment,
            "RSI": rsi,
            "MACD": macd,
            "MA50": ma50,
            "MA200": ma200,
            "Hist": hist
        }
    except Exception as e:
        return {"Kode": stock_code, "Error": str(e)}

# Streamlit layout
st.set_page_config(layout="wide")
st.title("Aplikasi Analisis Saham Ritel")

st.header("1. Portofolio Saham Saat Ini")
portfolio = load_portfolio()
st.dataframe(portfolio)

st.subheader("Edit Portofolio")
st.write("Tambahkan atau ubah portofolio di bawah ini:")
edited = st.data_editor(portfolio, num_rows="dynamic")
if st.button("Simpan Portofolio"):
    update_portfolio(edited)
    st.success("Portofolio diperbarui.")

st.header("2. Analisis Mendetail Saham")
analisis = []
for kode in portfolio['Kode']:
    result = analyze_stock(kode)
    if "Error" not in result:
        analisis.append(result)
analisis_df = pd.DataFrame(analisis)
st.dataframe(analisis_df)

st.header("3. Strategi Penambahan Saham")
modal = st.number_input("Masukkan modal tambahan (Rp):", min_value=0, step=100000)
if modal > 0:
    undervalued = analisis_df[analisis_df['Komentar_AI'] == 'Undervalued']
    if not undervalued.empty:
        harga = undervalued['Harga_Terakhir'] * 100
        porsi = modal * (1 / harga) / (1 / harga).sum()
        undervalued['Alokasi_Rp'] = porsi
        st.write("Rekomendasi alokasi modal:")
        st.dataframe(undervalued[['Kode', 'Harga_Terakhir', 'Alokasi_Rp']])
    else:
        st.warning("Tidak ada saham yang undervalued saat ini.")

st.header("4. Simulasi Pertumbuhan Bunga Majemuk")
compound_growth = st.slider("Estimasi pertumbuhan tahunan (%):", 5, 20, 10)

if not analisis_df.empty and 'Dividend_Yield' in analisis_df.columns:
    div_yield_avg = analisis_df['Dividend_Yield'].dropna().mean() / 100
else:
    div_yield_avg = 0.02  # fallback default 2% jika data tidak tersedia

start_value = (portfolio['Lot'] * portfolio['Harga_Beli'] * 100).sum()

def project_growth(initial, rate, years):
    return [initial * ((1 + rate) ** y) for y in years]

years = [3, 5, 7, 10]
rates = [((compound_growth + div_yield_avg * 100)/100)] * 4
values = project_growth(start_value, rates[0], years)

growth_df = pd.DataFrame({
    "Tahun": years,
    "Estimasi_Portofolio (Rp)": values
})
st.line_chart(growth_df.set_index("Tahun"))
st.dataframe(growth_df)

st.caption("*Data berdasarkan estimasi dan informasi dari Yahoo Finance.")
