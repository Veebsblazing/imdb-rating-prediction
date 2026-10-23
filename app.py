# app.py
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ---------- Load artefak saat startup ----------
print("Loading model artifacts...")
model = joblib.load("model/model.pkl")
mlb   = joblib.load("model/genre_encoder.pkl")
with open("model/meta.json") as f:
    META = json.load(f)

FEATURE_COLS   = META["feature_cols"]
GENRES         = META["genres"]
GROSS_MEAN     = META["gross_mean"]
META_SCORE_MED = META["meta_score_median"]
YEAR_MED       = META["released_year_median"]
print(f"Model siap. {len(FEATURE_COLS)} fitur | {len(GENRES)} genre")


# ---------- Endpoint root ----------
@app.route("/", methods=["GET"])
def home():
    return render_template("/index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/genres", methods=["GET"])
def list_genres():
    return jsonify({"count": len(GENRES), "genres": GENRES})


# ---------- Endpoint prediksi ----------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "Body harus JSON valid"}), 400

        # ---- Ambil & validasi field wajib ----
        runtime       = float(data.get("runtime", 120))
        released_year = float(data.get("released_year", YEAR_MED))
        meta_score    = float(data.get("meta_score", META_SCORE_MED))
        gross         = float(data.get("gross", GROSS_MEAN))

        genres_in = data.get("genres", [])
        if not isinstance(genres_in, list):
            return jsonify({"error": "'genres' harus list string"}), 400

        # Normalisasi nama genre (case-insensitive, mis. "sci-fi" -> "Sci-Fi")
        genre_map = {g.lower(): g for g in GENRES}
        normalized = []
        unknown = []
        for g in genres_in:
            key = str(g).lower()
            if key in genre_map:
                normalized.append(genre_map[key])
            else:
                unknown.append(g)

        if unknown:
            return jsonify({
                "error": f"Genre tidak dikenal: {unknown}",
                "hint": "Lihat GET /genres untuk daftar valid"
            }), 400

        if len(normalized) == 0:
            return jsonify({"error": "Minimal 1 genre valid diperlukan"}), 400

        # ---- One-hot encode genre ----
        genre_vec = mlb.transform([normalized])[0]
        genre_dict = {f"Genre_{g}": int(v) for g, v in zip(GENRES, genre_vec)}

        # ---- Susun vektor fitur SESUAI URUTAN training ----
        row = {
            "Runtime": runtime,
            "Released_Year": released_year,
            "Meta_score": meta_score,
            "Gross": gross,
            **genre_dict,
        }
        X = np.array([[row[c] for c in FEATURE_COLS]], dtype=float)

        # ---- Prediksi ----
        pred = float(model.predict(X)[0])

        return jsonify({
            "predicted_imdb_rating": round(pred, 3),
            "input": {
                "runtime": runtime,
                "released_year": int(released_year),
                "meta_score": meta_score,
                "gross": gross,
                "genres": normalized,
            },
            "note": "Prediksi berbasis Random Forest; RMSE test ≈ 0.23"
        })

    except ValueError as e:
        return jsonify({"error": f"Input tidak valid: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Internal error: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    
    from flask import Flask, request, jsonify, render_template 