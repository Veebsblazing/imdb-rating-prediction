# model/train_model.py
import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestRegressor

URL = "https://raw.githubusercontent.com/lilswapnil/bert-imdb-scraper/main/notebook/data/imdb_top_1000.csv"

print("1. Load dataset...")
df = pd.read_csv(URL)
print(f"   Shape awal: {df.shape}")

print("2. Cleaning (sesuai notebook)...")
df['Runtime'] = df['Runtime'].str.replace(' min', '', regex=False).astype(float)

df['Gross'] = df['Gross'].astype(str).str.replace(',', '', regex=False)
df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')
gross_mean = df['Gross'].mean()          # <-- SIMPAN untuk API
df['Gross'] = df['Gross'].fillna(gross_mean)

df['Meta_score'] = df['Meta_score'].fillna(df['Meta_score'].median())

df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
df['Released_Year'] = df['Released_Year'].fillna(df['Released_Year'].median())

df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
df = df.dropna(subset=['IMDB_Rating'])
print(f"   Shape setelah cleaning: {df.shape}")

print("3. Encoding genre (MultiLabelBinarizer)...")
df['Genre'] = df['Genre'].fillna('Unknown')
df['Genre_list'] = df['Genre'].str.split(', ')

mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(
    mlb.fit_transform(df['Genre_list']),
    columns=[f"Genre_{g}" for g in mlb.classes_]
)
print(f"   Jumlah genre unik: {len(mlb.classes_)}")

print("4. Susun X dan y...")
numeric_cols = ['Runtime', 'Released_Year', 'Meta_score', 'Gross']
X = pd.concat([
    df[numeric_cols].reset_index(drop=True),
    genre_encoded.reset_index(drop=True)
], axis=1)
y = df['IMDB_Rating'].reset_index(drop=True)
feature_cols = X.columns.tolist()
print(f"   Shape X: {X.shape}, y: {y.shape}")

print("5. Train-test split & training Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluasi cepat
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = rf.predict(X_test)
print(f"   RMSE: {mean_squared_error(y_test, y_pred)**0.5:.4f}")
print(f"   MAE : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"   R²  : {r2_score(y_test, y_pred):.4f}")

print("6. Simpan artefak ke folder model/...")
os.makedirs("model", exist_ok=True)
joblib.dump(rf,  "model/model.pkl")
joblib.dump(mlb, "model/genre_encoder.pkl")

meta = {
    "feature_cols": feature_cols,
    "genres": list(mlb.classes_),
    "gross_mean": float(gross_mean),
    "meta_score_median": float(df['Meta_score'].median()),
    "released_year_median": float(df['Released_Year'].median()),
    "model_type": "RandomForestRegressor(n_estimators=100, random_state=42)",
    "metrics": {
        "rmse": float(mean_squared_error(y_test, y_pred)**0.5),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
}
with open("model/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Selesai. Artefak tersimpan di folder model/")