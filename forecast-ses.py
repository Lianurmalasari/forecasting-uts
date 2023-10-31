import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Muat model Simple Exp Smoothing dari file .sav
model_file = 'forecast-ses.sav'
model = pickle.load(open(model_file, 'rb'))

# Muat dataset avocado.csv
data_file = 'avocado.csv'
data = pd.read_csv(data_file)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Resample DataFrame ke frekuensi mingguan
data = data.resample('W').sum()

# Judul aplikasi
st.title('Forecasting Rata-Rata Harga Alpukat')

# Slider untuk menentukan jumlah minggu yang akan diprediksi
forecast_steps = st.slider('Jumlah Minggu Prediksi', 1, 30, 12)

# Tombol "Prediksi"
if st.button('Prediksi'):
    # Prediksi dengan model Simple Exp Smoothing
    forecast = model.forecast(steps=forecast_steps)
    
    # Tampilkan data asli
    st.subheader('Data Asli')
    st.line_chart(data['AveragePrice'])  # Select the 'AveragePrice' column for the line chart

    # Tampilkan hasil prediksi
    st.subheader('Hasil Prediksi')
    st.line_chart(forecast)
    
    # Tampilkan tabel hasil prediksi
    st.subheader('Tabel Hasil Prediksi')
    forecast_df = pd.DataFrame({
    'Tanggal': pd.date_range(start=data.index[-1], periods=forecast_steps, freq='W'),
    'Prediksi': forecast
    })
    st.dataframe(forecast_df)
