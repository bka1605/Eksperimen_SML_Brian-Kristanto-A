import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_preprocessing():
    print("Memulai otomatisasi preprocessing data Mobile Price Classification...")

    # load data
    df = pd.read_csv('mobile_price_cls_raw/mobile_price_cls_train.csv')

    # cleaning data
    df = df.dropna()
    df = df.drop_duplicates()

    # pisahin x dan y
    X = df.drop(columns=['price_range'])
    y = df['price_range']

    # data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # scaling data
    scaler = StandardScaler()
    
    # ubah hasil scaling ke DataFrame biar bisa disimpan ke CSV 
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    y_train_df = pd.DataFrame(y_train, columns=['price_range'])
    y_test_df = pd.DataFrame(y_test, columns=['price_range'])

    # simpan ke folder
    os.makedirs('mobile_price_cls_preprocessing', exist_ok=True)
    
    X_train_scaled.to_csv('mobile_price_cls_preprocessing/X_train.csv', index=False)
    X_test_scaled.to_csv('mobile_price_cls_preprocessing/X_test.csv', index=False)
    y_train_df.to_csv('mobile_price_cls_preprocessing/y_train.csv', index=False)
    y_test_df.to_csv('mobile_price_cls_preprocessing/y_test.csv', index=False)

    print("Otomatisasi sukses! Data siap latih telah disimpan di folder 'mobile_price_cls_preprocessing'.")

if __name__ == "__main__":
    run_preprocessing()