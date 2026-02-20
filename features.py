# =========================================================
# features.py – Feature engineering and fatigue scoring
# =========================================================

import pandas as pd
import numpy as np

def engineer_features(df, window=5.0):
    print("Building rolling features...")
    df = df.sort_values(['player_api_id','date','match_minute']).reset_index(drop=True)
    result = []
    for pid, g in df.groupby('player_api_id'):
        g_roll = g.rolling(window=int(window), on='match_minute').agg({
            'speed':'mean',
            'sprint_count':'sum',
            'reaction_delay':'mean',
            'error_rate':'mean',
            'distance':'sum'
        })
        g_roll.columns = [f'{c}_{int(window)}m' for c in g_roll.columns]
        g_roll['player_api_id'] = pid
        g_roll['date'] = g['date'].values
        g_roll['match_minute'] = g['match_minute'].values
        result.append(g_roll)
    out = pd.concat(result, ignore_index=True).dropna()
    print("Features built successfully.")
    return out


def normalize_and_score(df):
    print("Normalizing and scoring fatigue...")
    cols = [c for c in df.columns if c.endswith('m')]
    out = df.copy()
    for c in cols:
        out[c+'_z'] = out.groupby('player_api_id')[c].transform(lambda x:(x-x.mean())/(x.std()+1e-6))

    out['physical_fatigue'] = (
        -out['speed_5m_z']*0.4 - out['sprint_count_5m_z']*0.3 - out['distance_5m_z']*0.3
    )

    out['mental_fatigue'] = (
        out['reaction_delay_5m_z']*0.6 + out['error_rate_5m_z']*0.4
    )

    for c in ['physical_fatigue','mental_fatigue']:
        out[c] = out.groupby('player_api_id')[c].transform(lambda x:(x-x.min())/(x.max()-x.min()+1e-9))

    out['combined_fatigue'] = 0.5*out['physical_fatigue'] + 0.5*out['mental_fatigue']
    print("Fatigue metrics computed.")
    return out
