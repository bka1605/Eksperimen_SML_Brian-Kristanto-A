import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_preprocessing():
    print("Memulai otomatisasi preprocessing data Iris...")

    # 1. Data Loading
    # Pastikan file iris.csv ada di dalam folder iris_raw
    df = pd.read_csv('iris_raw/iris.csv')

    # 2. Pembersihan Data (Menghapus Missing Values & Duplicates)
    df = df.dropna()
    df = df.drop_duplicates()

    # 3. Memisahkan Fitur (X) dan Target (y)
    X = df.drop(columns=['target'])
    y = df['target']

    # 4. Splitting Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Scaling Fitur menggunakan StandardScaler
    scaler = StandardScaler()
    
    # Konversi kembali output scaler menjadi Pandas DataFrame agar bisa menggunakan .to_csv
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    y_train_df = pd.DataFrame(y_train, columns=['target'])
    y_test_df = pd.DataFrame(y_test, columns=['target'])

    # 6. Menyimpan Hasil Preprocessing ke Folder
    os.makedirs('iris_preprocessing', exist_ok=True)
    
    X_train_scaled.to_csv('iris_preprocessing/X_train.csv', index=False)
    X_test_scaled.to_csv('iris_preprocessing/X_test.csv', index=False)
    y_train_df.to_csv('iris_preprocessing/y_train.csv', index=False)
    y_test_df.to_csv('iris_preprocessing/y_test.csv', index=False)

    print("Otomatisasi sukses! Data siap latih telah disimpan di folder 'iris_preprocessing'.")

if __name__ == "__main__":
    run_preprocessing()