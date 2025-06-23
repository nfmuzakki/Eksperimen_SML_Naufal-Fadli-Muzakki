# Library dasar python
import pandas as pd
import os
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Library untuk preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_column, save_path, file_path):
    # Penanganan outlier dengan IQR
    for col in df.drop(columns=target_column).columns:
        q1=np.quantile(df[col],0.25)
        q3=np.quantile(df[col],0.75)
        iqr=q3-q1
        lb=q1-iqr*1.5
        ub=q3+iqr*1.5
        df=df[(df[col]>=lb)&(df[col]<=ub)]
    
    # Mendapatkan nama kolom tanpa kolom target
    column_names = df.columns.drop(target_column)
 
    # Membuat DataFrame kosong dengan nama kolom
    df_header = pd.DataFrame(columns=column_names)
 
    # Menyimpan nama kolom sebagai header tanpa data
    df_header.to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    # Encode variabel target menggunakan label encoder
    le=LabelEncoder()
    df[target_column]=le.fit_transform(df[target_column])

    # Simpan encoder
    joblib.dump(le, save_path)

    # Membagi data menjadi train dan test
    x=df.drop(columns=target_column)
    y=df[target_column]
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

    # Simpan data preprocessed
    os.makedirs("preprocessing/dataset_preprocessing", exist_ok=True)
    X_train.to_csv("preprocessing/dataset_preprocessing/X_train.csv", index=False)
    X_test.to_csv("preprocessing/dataset_preprocessing/X_test.csv", index=False)
    Y_train.to_csv("preprocessing/dataset_preprocessing/Y_train.csv", index=False)
    Y_test.to_csv("preprocessing/dataset_preprocessing/Y_test.csv", index=False)

    return X_train, X_test, Y_train, Y_test

if __name__ == "__main__":
    df = pd.read_csv("pollution_dataset_raw.csv")
    preprocess_data(df, "Air Quality", "preprocessing/preprocessor_le.joblib", "preprocessing/dataset.csv")