# =========================================================
# data_prep.py – Extract data and simulate minute-wise stats
# =========================================================

import sqlite3
import pandas as pd
import numpy as np

def load_data(db_path="database.sqlite"):
    print("Connecting to database...")
    conn = sqlite3.connect(db_path)

    player_attr = pd.read_sql_query("""
        SELECT player_api_id,
               date,
               overall_rating,
               stamina,
               strength,
               vision,
               agility AS composure
        FROM Player_Attributes
        WHERE overall_rating IS NOT NULL
    """, conn)

    conn.close()
    print("Data loaded successfully.")

    np.random.seed(42)
    records = []
    for _, row in player_attr.iterrows():
        # Simulate 90 minutes of match data per record
        for minute in range(0, 91, 5):  # every 5-minute interval
            fatigue_factor = np.exp(minute / 90) / np.e  # grows with time
            records.append({
                'player_api_id': row['player_api_id'],
                'date': row['date'],
                'match_minute': minute,
                'speed': (row['stamina'] / 10) - 0.05 * fatigue_factor + np.random.uniform(-0.2, 0.2),
                'sprint_count': int(row['stamina'] / 12 - 0.02 * fatigue_factor * 10 + np.random.uniform(0, 2)),
                'reaction_delay': 1.2 - (row['vision'] / 100) + 0.01 * fatigue_factor + np.random.normal(0, 0.05),
                'error_rate': (100 - row['composure']) / 100 + 0.005 * fatigue_factor + np.random.normal(0, 0.01),
                'distance': row['stamina'] * 0.5 * (minute / 90) + np.random.uniform(0, 5)
            })
    df = pd.DataFrame(records)
    print(f"Generated simulated minute-wise data → {len(df)} rows.")
    return df
