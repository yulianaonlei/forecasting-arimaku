import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import timedelta

st.set_page_config(page_title="Forecasting ARIMA Sparepart & Transaksi", layout="wide")
st.title("Forecasting ARIMA - Nilai Transaksi & Jumlah Sparepart Terjual")

# CSS
st.markdown("""
<style>
    body { background-color: #f8f9fa; }
    .main { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
    h1, h2, h3, h4 { color: #2c3e50; }
    .stButton>button { background-color: #4CAF50; color: white; padding: 0.5em 1em; border-radius: 8px; border: none;}
    .stButton>button:hover { background-color: #45a049; }
    .stDownloadButton>button { background-color: #3498db; color: white; border-radius: 8px;}
    .stDownloadButton>button:hover { background-color: #2980b9; }
</style>
""", unsafe_allow_html=True)

# Upload file Excel
uploaded_file = st.file_uploader("📂 Upload file Excel (.xlsx/.xls)", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    date_column = st.selectbox("🗓️ Pilih kolom tanggal:", df.columns)
    df[date_column] = pd.to_datetime(df[date_column])

    col_value = st.selectbox("💰 Pilih kolom Nilai Transaksi:", df.columns)
    col_qty = st.selectbox("🛠️ Pilih kolom Jumlah Sparepart Terjual:", df.columns)

    # Pastikan numerik
    df[col_value] = pd.to_numeric(df[col_value], errors='coerce').fillna(0)
    df[col_qty] = pd.to_numeric(df[col_qty], errors='coerce').fillna(0)

    # Agregasi per hari
    daily_value = df.groupby(date_column)[col_value].sum().sort_index()
    daily_qty = df.groupby(date_column)[col_qty].sum().sort_index()

    target_series_name = st.selectbox("🔍 Pilih data untuk uji stasioneritas & plot:", ['Nilai Transaksi', 'Jumlah Sparepart Terjual'])
    target_series = daily_value if target_series_name == 'Nilai Transaksi' else daily_qty

    # Uji stasioneritas
    st.subheader("✅ Uji Stasioneritas (ADF Test)")
    result_adf = adfuller(target_series.dropna())
    p_value = result_adf[1]
    st.write(f"**ADF Statistic:** {result_adf[0]:.4f}")
    st.write(f"**p-value:** {p_value:.4f}")

    # Plot before-after differencing
    st.subheader("📊 Plot Sebelum & Sesudah Differencing")
    fig_diff, ax_diff = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    ax_diff[0].plot(target_series, color='blue')
    ax_diff[0].set_title('Original Series')

    if p_value > 0.05:
        differenced_series = target_series.diff().dropna()
        ax_diff[1].plot(differenced_series, color='green')
        ax_diff[1].set_title('After Differencing (d=1)')
        st.warning("Data cenderung *non-stationary* (p-value > 0.05). Plot dan uji berikut memakai data differencing (d=1).")
        stationary_series = differenced_series
    else:
        ax_diff[1].plot(target_series, color='green')
        ax_diff[1].set_title('Already Stationary')
        st.success("Data sudah *stationary* (p-value <= 0.05).")
        stationary_series = target_series

    st.pyplot(fig_diff)

    # Plot ACF & PACF
    st.subheader("📊 Plot ACF & PACF (data yang sudah stationary)")
    fig_acf, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(stationary_series, ax=axes[0], lags=30)
    axes[0].set_title('ACF')
    plot_pacf(stationary_series, ax=axes[1], lags=30, method='ywm')
    axes[1].set_title('PACF')
    st.pyplot(fig_acf)

    # Parameter ARIMA manual
    st.subheader("⚙️ Parameter ARIMA")
    p = st.number_input("p (ordo AR)", min_value=0, step=1, value=2)
    d = st.number_input("d (ordo Differencing)", min_value=0, step=1, value=1)
    q = st.number_input("q (ordo MA)", min_value=0, step=1, value=0)

    forecast_days = int(st.number_input("📅 Jumlah hari peramalan:", min_value=1, step=1, value=7))

    if st.button("FORECASTING"):
        last_date = daily_value.index.max()
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days+1)]

        def forecast_arima(series, p, d, q, steps):
            model = ARIMA(series, order=(p,d,q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            return forecast

        forecast_value = forecast_arima(daily_value, p, d, q, forecast_days)
        forecast_qty = forecast_arima(daily_qty, p, d, q, forecast_days)

        result = pd.DataFrame({
            "Tanggal": forecast_dates,
            "Forecast Nilai Transaksi": forecast_value,
            "Forecast Jumlah Sparepart Terjual": forecast_qty
        })

        st.subheader("📅 Hasil Forecasting")
        st.write(result)

        # Visualisasi forecast
        st.subheader("📈 Visualisasi Forecasting")
        fig2, ax2 = plt.subplots(figsize=(10,5))
        ax2.plot(daily_value.index, daily_value, label='Aktual Nilai Transaksi', color='blue')
        ax2.plot(forecast_dates, forecast_value, label='Forecast Nilai Transaksi', color='orange', linestyle='--')
        ax2.legend()
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(10,5))
        ax3.plot(daily_qty.index, daily_qty, label='Aktual Sparepart Terjual', color='green')
        ax3.plot(forecast_dates, forecast_qty, label='Forecast Sparepart Terjual', color='red', linestyle='--')
        ax3.legend()
        st.pyplot(fig3)

        # Download hasil forecast
        csv_data = result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download hasil forecasting (.csv)",
            data=csv_data,
            file_name='forecast_result.csv',
            mime='text/csv'
        )

        # Footer
        st.markdown("""
        <hr style="margin-top: 40px; margin-bottom:10px;">
        <div style="text-align: center; color: #888;">
            STMIK El Rahma Yogyakarta | Yuliana 12211870 ❤️
        </div>
        """, unsafe_allow_html=True)
