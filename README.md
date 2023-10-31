# Laporan Proyek Machine Learning
### Nama : Lia Nurmalasari
### Nim : 211351073
### Kelas : IF Malam B

## Domain Proyek

Pembuatan sistem forecasting rata-rata harga alpukat ini agar mempermudah penjual dan pembeli untuk mengetahui perubahan harga alpukat berdasarkan waktu yaitu perminggu.

## Business Understanding

Forecasting rata-rata harga alpukat ini dibuat agar penjual alpukat mendapatkan informasi kapan harga rata-rata alpukat akan tinggi dan pembeli dapat mengetahui rata-rata harga alpukat agar dapat membeli alpukat dengan harga yang tidak terlalu tinggi tetapi tetap pada rata-rata yang sudah di prediksi

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Harga alpukat yang berubah seiring dengan waktu
- Harga alpukat yang berbeda tiap tokonya
- Pembeli yang harus cek harga pertoko agar tau rata-rata harga alpukat

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mempermudah untuk mengetahui perubahan rata-rata harga alpukat
- Mempermudah untuk mengetahui rata-rata harga alpukat
- Pembeli tidak perlu datang ke setiap toko untuk tau hara rata-rata alpukat

    ### Solution statements
    - Membuat sistem yang memudahkan setiap orang dalam mendapatkan informasi tentang rata-rata harga alpukat tanpa perlu datang ke ke tokonya
    - Sistem yang dibuat menggunakan dataset yang diambil dari kaggle dan diproses menggunakan 3 algoritma yaitu ARIMA, Single Exponential Smoothing dan Double Exponential Smoothing yang mana selanjutnya akan dipilih algoritma terbaik dengan nilai RMSE terkecil untuk dipakai didalam sistem tersebut

## Data Understanding
Dataset yang diambil dari kaggle ini berisi 9 kolom dan yang dipakai hanyalah 2 yaitu tanggal dan Rata-rata harga alpukat. 

Dataset: [Avocado Prices](https://www.kaggle.com/datasets/neuromusic/avocado-prices)).

Dalam proses data understanding ini tahapan pertama yang dilakukan adalah:
1. import dataset

dikarenakan dataset diambil dari kaggle maka kita perlu import token kaggle kita:
```
from google.colab import files
files.upload()
```
lalu kita buat directory nya:
```
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
selanjutnya download datasetnya:
```
!kaggle datasets download -d neuromusic/avocado-prices
```
setelah itu kita unzip file yang sudah di download:
```
!mkdir avocado-prices
!unzip avocado-prices.zip -d avocado-prices
!ls avocado-prices
```
jangan lupa untuk import library yang akan digunakan:
```
import pandas as pd
import numpy as np

# library untuk lvisualisasi
import matplotlib.pyplot as plt
import seaborn as sns

# library untuk analisis time series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# library yang digunakan untuk forecasting
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
```
Setelah itu baru kita import datasetnya:
```
df = pd.read_csv("/content/avocado-prices/avocado.csv")
```
2. menampilkan 5 baris pertama dataset
```
df.head()
```
3. cek tipe data
```
df.info()
```
<img width="240" alt="image" src="https://github.com/Lianurmalasari/forecasting-uts/assets/145843965/d227d514-dfc3-4686-8f74-5cbd25ddd997">

4. cek ukuran dataset
```
df.shape
```
(18249, 14)

5. Null Check
```
df.isnull().sum()
```
Tidak terdapat data yang kosong

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- Date - Tanggal pengamatan
- AveragePrice - harga rata-rata satu buah alpukat
- type - konvensional atau organik
- year - tahun
- Region - kota atau wilayah pengamatan
- Total Volume - Jumlah total alpukat yang terjual
- 4046 - Jumlah total alpukat dengan PLU 4046 yang terjual
- 4225 - Jumlah total alpukat dengan PLU 4225 yang terjual
- 4770 - Jumlah total alpukat dengan PLU 4770 yang terjual

## Data Preparation
hapus kolom yang tidak akan di pakai:
```
df = df.drop(['Unnamed: 0','Total Volume', '4046', '4225','4770', 'Total Bags',
              'Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year', 'region'], axis=1)
```
Dikarenakan kolom Month memiliki tipe data object maka harus kita convert menjadi datetime:
```
df['Date'] = pd.to_datetime(df['Date'])
```
lalu kita set index dari kolom month untuk menjadi acuan dalam melakukan forecasting:
```
df.set_index("Date", inplace=True)
```
kita buat resample data sekaligus setting frequensinya:
```
df = df.resample('W').sum()
```

kita tampilkan juga bagaimana grafik dari perubahan jumlah penumpang pasawatnya:
```
df['AveragePrice'].plot(figsize=(12,5));
```
![image](https://github.com/Lianurmalasari/forecasting-uts/assets/145843965/7134ec75-5bc3-4feb-8daa-8a35508d4f91)

Selanjutnya kita bagi terlebih dahulu antara train dan test data:
```
train = df.iloc[:100]
test = df.iloc[101:]
```

Sekarang mari kita lihat selisih antara setiap dua poin data berturut-turut dalam sebuah rangkaian data deret waktu (time series):
```
diff_df = df.diff()
diff_df.head()
```
lalu kita hapus kolom yang berisi null values:
```
diff_df.dropna(inplace=True)
```
Sekarang kita lakukan uji adfuller yaitu uji statistik yang digunakan untuk mengevaluasi apakah sebuah deret waktu stasioner atau tidak:
```
result = adfuller(diff_df)
# The result is a tuple that contains various test statistics and p-values
# You can access specific values as follows:
adf_statistic = result[0]
p_value = result[1]

# Print the results
print(f'ADF Statistic: {adf_statistic}')
print(f'p-value: {p_value}')
```
Selanjutnya kita cek korelasi dari deret waktunya:
```
plot_acf(diff_df)
plot_pacf(diff_df)
```
![image](https://github.com/Lianurmalasari/forecasting-uts/assets/145843965/3c3cb07c-0b36-4c51-925e-b8464feacab2)


## Modeling
Ditahap modeling ini kita akan menggunakan 3 algoritma yang mana akan kita bandingkan algoritma terbaik yang selanjutnya akan dipakai untuk aplikasi tersebut.

kita akan coba untuk memprediksi 43 bulan kedepan:

  ### Single Exponential Smoothing
```
single_exp = SimpleExpSmoothing(train).fit()
single_exp_train_pred = single_exp.fittedvalues
single_exp_test_pred = single_exp.forecast(68)
```
```
train['AveragePrice'].plot(style='--', color='gray', legend=True, label='train')
test['AveragePrice'].plot(style='--', color='r', legend=True, label='test')
single_exp_test_pred.plot(color='b', legend=True, label='Prediction')
```
![image](https://github.com/Lianurmalasari/forecasting-uts/assets/145843965/020631a0-87a9-4b16-9a19-49d180470f18)

```
Train_RMSE_SES = mean_squared_error(train, single_exp_train_pred)**0.5
Test_RMSE_SES = mean_squared_error(test, single_exp_test_pred)**0.5
Train_MAPE_SES = mean_absolute_percentage_error(train, single_exp_train_pred)
Test_MAPE_SES = mean_absolute_percentage_error(test, single_exp_test_pred)

print('Train RMSE :',Train_RMSE_SES)
print('Test RMSE :', Test_RMSE_SES)
print('Train MAPE :', Train_MAPE_SES)
print('Test MAPE :', Test_MAPE_SES)
```
Train RMSE : 5.153814386780101<br>
Test RMSE : 19.64903748252027<br>
Train MAPE : 0.026191870205027584<br>
Test MAPE : 0.10254968128586632<br>

  ## Double Exponential Smoothing
```
double_exp = ExponentialSmoothing(train, trend=None, initialization_method='heuristic', seasonal='add', seasonal_periods=29, damped_trend=False).fit()
double_exp_train_pred = double_exp.fittedvalues
double_exp_test_pred = double_exp.forecast(68)
```
```
train['AveragePrice'].plot(style='--', color='gray', legend=True, label='train')
test['AveragePrice'].plot(style='--', color='r', legend=True, label='test')
double_exp_test_pred.plot(color='b', legend=True, label='Prediction')
```
![image](https://github.com/Lianurmalasari/forecasting-uts/assets/145843965/7d80dd53-3a55-4419-a1aa-0ae6d344565b)


```
Train_RMSE_DES = mean_squared_error(train, double_exp_train_pred)**0.5
Test_RMSE_DES = mean_squared_error(test, double_exp_test_pred)**0.5
Train_MAPE_DES = mean_absolute_percentage_error(train, double_exp_train_pred)
Test_MAPE_DES = mean_absolute_percentage_error(test, double_exp_test_pred)

print('Train RMSE :',Train_RMSE_DES)
print('Test RMSE :', Test_RMSE_DES)
print('Train MAPE :', Train_MAPE_DES)
print('Test MAPE :', Test_MAPE_DES)
```
Train RMSE : 5.008014846943417<br>
Test RMSE : 20.659434661969186<br>
Train MAPE : 0.02371272497473234<br>
Test MAPE : 0.11139236252098973<br>

  ## ARIMA
```
ar = ARIMA(train, order=(15,1,15)).fit()
ar_train_pred = ar.fittedvalues
ar_test_pred = ar.forecast(68)
```
```
train['AveragePrice'].plot(style='--', color='gray', legend=True, label='train')
test['AveragePrice'].plot(style='--', color='r', legend=True, label='test')
ar_test_pred.plot(color='b', legend=True, label='Prediction')
```
![image](https://github.com/Lianurmalasari/forecasting-uts/assets/145843965/f79cbc20-cedf-472c-abb9-fa6fe6c0bc0e)

```
Train_RMSE_AR = mean_squared_error(train, ar_train_pred)**0.5
Test_RMSE_AR = mean_squared_error(test, ar_test_pred)**0.5
Train_MAPE_AR = mean_absolute_percentage_error(train, ar_train_pred)
Test_MAPE_AR = mean_absolute_percentage_error(test, ar_test_pred)

print('Train RMSE :',Train_RMSE_AR)
print('Test RMSE :', Test_RMSE_AR)
print('Train MAPE :', Train_MAPE_AR)
print('Test MAPE :', Test_MAPE_AR)
```
Train RMSE : 14.635330876406917<br>
Test RMSE : 20.204805696354683<br>
Train MAPE : 0.03223810862444207<br>
Test MAPE : 0.10467032195997751<br>

Selanjutnya mari kita evaluasi 3 algoritma tersebut

## Evaluation
Pada tahap evaluasi ini kita akan membandingkan nilai Root Mean Squared Error (RMSE) dan Mean Absolute Percentage Error (MAPE) yang mana selanjutnya akan kita urutkan mana nilai RMSE yang paling kecil maka algoritma tersebut yang akan kita pakai:
```
comparision_df = pd.DataFrame(data=[
    ['Single Exp Smoothing', Test_RMSE_SES, Test_MAPE_SES],
    ['Double Exp Smoothing', Test_RMSE_DES, Test_MAPE_DES],
    ['ARIMA', Test_RMSE_AR, Test_MAPE_AR]
    ],
    columns=['Model', 'RMSE', 'MAPE'])
comparision_df.set_index('Model', inplace=True)
```
```
comparision_df.sort_values(by='RMSE')
```
<img width="227" alt="image" src="https://github.com/Lianurmalasari/forecasting-uts/assets/145843965/b1efaf61-dae3-4c3a-8b7d-feb388618f1a">

dapat dilihat jika nilai RMSE dan MAPE ada pada algoritma Single Exponential Smoothing, maka dari itu algoritma yang akan dipakai adalah algoritma Single Exponential Smoothing

coba prediksi menggunakan algoritma Single Exponential Smoothing:
```
single_exp = SimpleExpSmoothing(df).fit()
single_exp_test_pred = single_exp.forecast(68)
```
```
df['AveragePrice'].plot(style='--', color='gray', legend=True, label='known')
single_exp_test_pred.plot(color='b', legend=True, label='Prediction')
```
![image](https://github.com/Lianurmalasari/forecasting-uts/assets/145843965/f7279c6b-2fe2-4c70-90ea-4df4632f8d69)


## Deployment
Link Aplikasi: [Forecasting Rata-Rata Harga Alpukat](https://forecast-alpukat-lia.streamlit.app/)

