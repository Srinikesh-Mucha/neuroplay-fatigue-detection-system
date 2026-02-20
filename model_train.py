# =========================================================
# model_train.py – Train & predict fatigue per minute
# =========================================================

from data_prep import load_data
from features import engineer_features, normalize_and_score
from utils import plot_fatigue, export_latest_json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib
import os

# =========================================================
# STEP 1: Load and prepare data
# =========================================================
df = load_data("database.sqlite")
features = engineer_features(df, window=5)
scored = normalize_and_score(features)

# =========================================================
# STEP 2: Train predictive model
# =========================================================
print("Training fatigue prediction model...")
scored = scored.sort_values(['player_api_id','date','match_minute'])
scored['future_fatigue'] = scored.groupby('player_api_id')['combined_fatigue'].shift(-1)
scored = scored.dropna(subset=['future_fatigue'])

X = scored[['speed_5m_z','sprint_count_5m_z','reaction_delay_5m_z','error_rate_5m_z','distance_5m_z','match_minute']]
y = scored['future_fatigue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = xgb.XGBRegressor(
    n_estimators=300, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
model.fit(X_train, y_train, verbose=False)

preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"✅ Model trained successfully. Test RMSE: {rmse:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fatigue_xgb.joblib")
print("Model saved in /models folder.")

# =========================================================
# STEP 3: User interaction
# =========================================================
player_id = int(input("Enter Player ID: "))
minute = int(input("Enter Match Minute (0–90): "))

plot_fatigue(scored, player_id)
latest_json = export_latest_json(scored, player_id, minute)
if latest_json:
    print("\nPredicted fatigue at that time:")
    print(latest_json)

print("\n✅ All steps completed successfully!")
