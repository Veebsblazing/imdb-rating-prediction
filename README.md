# Proyek ML: Prediksi Rating IMDb Top 1000

## Deskripsi
Membangun model regresi untuk memprediksi rating film IMDb
berdasarkan runtime, tahun rilis, skor Metacritic, pendapatan kotor, dan genre.

## Dataset
- Sumber: https://github.com/lilswapnil/bert-imdb-scraper
- Jumlah: 1000 film × 16 kolom

## Metode
1. Data cleaning (Runtime, Gross, Meta_score)
2. Multi-label encoding untuk Genre
3. Train-test split 80:20
4. Model: Linear Regression, Random Forest, Gradient Boosting
5. Evaluasi: RMSE, MAE, R², Cross-Validation 5-fold

## Hasil
| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 0.2510 | 0.2108 | 0.0173 |
| Random Forest | 0.2314 | 0.1854 | 0.1650 |
| Gradient Boosting | 0.2380 | 0.1960 | 0.1165 |

## Cara Menjalankan
```bash
pip install -r requirements.txt
jupyter notebook Notebooks/analysis.ipynb

## Kesimpulan

1. Model terbaik berdasarkan RMSE adalah Random Forest dengan RMSE = 0.2314.
2. Fitur paling berpengaruh terhadap prediksi rating adalah 
   **Meta_score**, **Gross**, dan **Runtime**.
3. Scatter plot aktual vs prediksi menunjukkan model cukup baik dalam mengikuti
   tren, meskipun prediksi cenderung mengumpul di rentang 7.6–8.4.
4. Keterbatasan: dataset hanya 1000 film, tidak representatif untuk seluruh film IMDb.

## Author

**Nama:** RIZKY SAPUTRA
**NIM:** 2410140283
**Mata Kuliah:** Machine Learning
**Universitas:** Universitas Pamulang