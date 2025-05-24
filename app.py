import streamlit as st
import pandas as pd
from modules import (
    analisis_valuasi,
    grafik_harga,
    strategi_tambah,
    edit_portofolio,
    simulasi_pertumbuhan,
    indikator_jual,
    ai_valuasi
)

def main():
    st.set_page_config(page_title="Sistem Analisis Saham", layout="wide")
    st.title("Aplikasi Analisis Saham ala Sekuritas Besar")

    # Load / Edit Portofolio
    portofolio = edit_portofolio.load_portofolio()

    if portofolio.empty:
        st.warning("Silakan masukkan data saham terlebih dahulu.")
        return

    # Analisis Valuasi
    portofolio = analisis_valuasi.analyze(portofolio)

    # Tampilan Valuasi & Grafik
    st.subheader("Analisis Valuasi Saham")
    st.dataframe(portofolio)

    grafik_harga.display(portofolio)

    # AI Penilaian Umum
    ai_valuasi.display(portofolio)

    # Strategi Tambah Saham
    strategi_tambah.display(portofolio)

    # Simulasi Pertumbuhan
    simulasi_pertumbuhan.display(portofolio)

    # Indikator Jual / Tahan
    indikator_jual.display(portofolio)

if __name__ == "__main__":
    main()