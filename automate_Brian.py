import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Bikin folder output kalau belum ada
os.makedirs('iris_preprocessing', exist_ok=True)

# 2. Muat dataset mentah
print("Memuat dataset iris...")
df = pd.read_csv('iris_raw/iris.csv')

# 3. Pisahkan fitur (X) dan target (y)
X = df.drop(columns=['target'])
y = df['target']

# 4. Splitting Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scaling Fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Konversi kembali ke DataFrame agar gampang disimpan
X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# 6. Simpan hasil preprocessing ke CSV
X_train_df.to_csv('iris_preprocessing/X_train.csv', index=False)
X_test_df.to_csv('iris_preprocessing/X_test.csv', index=False)
y_train.to_csv('iris_preprocessing/y_train.csv', index=False)
y_test.to_csv('iris_preprocessing/y_test.csv', index=False)

print("Berhasil! Data yang sudah di-preprocess tersimpan di folder 'iris_preprocessing'.")