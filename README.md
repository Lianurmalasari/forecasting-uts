# Laporan Proyek Machine Learning
### Nama : Lia Nurmalasari
### Nim : 211351073
### Kelas : Malam B

## Domain Proyek

Untuk menjaga nilai ekonomi kacang pistachio yang memiliki tempat penting dalam ekonomi pertanian, efisiensi proses industri pascapanen menjadi sangat penting. Untuk memberikan efisiensi ini, diperlukan metode dan teknologi baru untuk pemisahan dan klasifikasi kacang pistachio. Spesies pistachio yang berbeda ditujukan untuk pasar yang berbeda, sehingga meningkatkan kebutuhan akan klasifikasi spesies pistachio. 

## Business Understanding

Dari banyaknya jenis kacang pistachio, sistem ini akan membantu dalam mengklasifikasi jenis pistachio diantara Kirmizi Pistachio atau Siit Pistachio

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Kurangnya efisiensi pemisahan dan klasifikasi kacang pistachio

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Dapat mempermudah dalam pemisahan dan klasifikasi kacang pistachio

    ### Solution statements
    - Pembuatan aplikasi yang dapat mengklasifikasi jenis pistachio dengan inputan karakteristik dari kacang tersebut yang dapat mempermudah proses pengklasifikasian dengan akurasi yang lebih dari 70%.
    - Model yang dipakai di aplikasi tersebut dibuat menggunakan algoritma SVM.

## Data Understanding
Dataset yang dipakai diambil dari kaggle yang berisi 2148 baris dan 17 kolom dimana jumlah atribut yang dipakai hanya 12 atribut sesuai dengan kebutuhan pembuatan model.

[Pistachio Dataset](https://www.kaggle.com/datasets/muratkokludataset/pistachio-dataset).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Dataset adalah sebagai berikut:
deskripsi tabel:

- AREA = luas pistachio
- PERIMETER = keliling pistachio
- MAJOR_AXIS = sumbu utama
- MINOR_AXIS = sumbu minor
- ECCENTRICITY = eksentrisitas
- EQDIASQ = luas sama dengan diameter persegi
- SOLIDITY = kepadatan pistachio
- CONVEX_AREA = luas cembung pistachio
- EXTENT = tingkat perluasan pistachio
- ASPECT_RATIO = rasio aspek pistachio
- ROUNDNESS = kebulatan pistachio
- COMPACTNESS = kepadatan pistachio
- SHAPEFACTOR_1 - SHAPEFACTOR_4 = Kompleksitas atau kerumitan betuk pistachio
- Class = Kelas/jenis pistachio (Kirmizi Pistachio & Siit Pistachio)

Dengan semua tipe data float kecuali Class yaitu object/string.

## Data Preparation
1. Library yang dipakai
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
```
2. Merubah tipe data object/string menjadi numerik(integer)
```
label_encoder = LabelEncoder()
df['Class'] = label_encoder.fit_transform(df['Class'])
```
3. Menghapus kolom yang tidak dipakai
```
columns_to_drop = ['SHAPEFACTOR_1', 'SHAPEFACTOR_2', 'SHAPEFACTOR_3', 'SHAPEFACTOR_4']
df = df.drop(columns=columns_to_drop)
```

## Modeling
1. Tentukan X dan Y
```
X = df.drop (columns='Class', axis=1)
Y = df['Class']
```
2. Scaling dataset
```
scaler = StandardScaler()
scaler.fit(X)
standarized_data = scaler.transform(X)
```
3. Tentukan X dan Y yang sudah di scaling
```
X = standarized_data
Y = df['Class']
```
4. Memisahkan data training dan testing
```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, stratify=Y, random_state=2)
```
5. Membuat model klasifikasi
```
classifier = svm.SVC(kernel='linear')
```
```
classifier.fit(X_train, Y_train)
```

## Evaluation

Pada tahan evaluasi, metrik akurasi adalah metode yang dipakai dalam pengujian model yang sudah dibuat:
```
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
```
```
print('Akurasi data training adalah = ', training_data_accuracy)
```
Akurasi data training adalah =  0.8760186263096624

```
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
```
```
print('Akurasi data testing adalah = ', test_data_accuracy)
```
Akurasi data testing adalah =  0.8697674418604651

Berdasarkan dari evaluasi yang sudah dilakukan, didapatkan akurasi data training sebesar 87% dan data testing sebesar 86% yang mana menunjukan jika model yang dibuat sudah bagus dan dapat dipakai.

## Deployment
[Link Aplikasi](https://klasifikasi-pistachio.streamlit.app/)

