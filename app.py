from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import os
from features import engineer_features, normalize_and_score
from data_prep import load_data
import numpy as np

app = Flask(__name__)

# ==============================
# Load data & model at startup
# ==============================
print("Connecting to database...")
df = load_data("database.sqlite")
features = engineer_features(df, window=5)
scored = normalize_and_score(features)
print("Data ready.")

# Ensure match_minute exists
if 'match_minute' not in scored.columns:
    scored['match_minute'] = np.random.randint(0,91,len(scored))

model_path = "models/fatigue_xgb.joblib"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Run model_train.py first.")

model = joblib.load(model_path)
print("Model loaded.")

# ==============================
# Routes
# ==============================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        player_id = int(request.form.get('player_id'))
        match_minute = int(request.form.get('match_minute'))

        row = scored[(scored['player_api_id']==player_id) & 
                     (scored['match_minute']==match_minute)]
        if row.empty:
            return jsonify({"error": "No data for that player/minute"}), 404

        X_row = row[['speed_5m_z','sprint_count_5m_z','reaction_delay_5m_z',
                     'error_rate_5m_z','distance_5m_z','match_minute']]
        pred_combined = float(model.predict(X_row)[0])

        # Ratio-based split
        physical_ratio = row['physical_fatigue'].values[0] / row['combined_fatigue'].values[0]
        mental_ratio = row['mental_fatigue'].values[0] / row['combined_fatigue'].values[0]

        pred_physical = pred_combined * physical_ratio
        pred_mental = pred_combined * mental_ratio

        return jsonify({
            "player_id": player_id,
            "match_minute": match_minute,
            "physical_fatigue": round(pred_physical,3),
            "mental_fatigue": round(pred_mental,3),
            "combined_fatigue": round(pred_combined,3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# Run app
# ==============================
if __name__ == '__main__':
    import webbrowser
    port = 5000
    webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(debug=True, port=port)
