# =========================================================
# utils.py – Visualization and JSON export (Improved)
# =========================================================

import json
import os
import pandas as pd
import plotly.express as px

def plot_fatigue(df, player_id, auto_open=True):
    """
    Saves an interactive fatigue plot for the selected player.
    Shows only 3 lines (physical, mental, combined).
    Opens it in the browser (optional).
    """
    player_df = df[df['player_api_id'] == player_id]
    if player_df.empty:
        print(f"⚠️ No data found for player {player_id}.")
        return

    # If there are multiple rows per minute, average them
    player_df = (
        player_df.groupby('match_minute', as_index=False)[
            ['physical_fatigue', 'mental_fatigue', 'combined_fatigue']
        ].mean()
    )

    # Melt into long format for Plotly
    df_melted = player_df.melt(
        id_vars='match_minute',
        value_vars=['physical_fatigue', 'mental_fatigue', 'combined_fatigue'],
        var_name='Fatigue Type',
        value_name='Fatigue Level'
    )

    # Create smooth line plot
    fig = px.line(
        df_melted,
        x='match_minute',
        y='Fatigue Level',
        color='Fatigue Type',
        title=f'Player {player_id} – Fatigue Progression During Match',
        labels={'match_minute': 'Match Minute'},
        line_shape='spline'
    )

    # Clean layout
    fig.update_layout(
        template='plotly_white',
        xaxis_title='Match Minute',
        yaxis_title='Fatigue Level',
        legend_title='Fatigue Type',
        font=dict(size=14)
    )

    # Ensure output folder exists
    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/player_{player_id}_fatigue.html"

    # Save interactive chart
    fig.write_html(plot_path, auto_open=auto_open)
    print(f"✅ Fatigue plot saved: {plot_path}")



def export_latest_json(df, player_id, minute, path="latest_prediction.json"):
    """
    Exports the fatigue prediction for a given player and minute to a JSON file.
    """
    latest = df[(df['player_api_id'] == player_id) & (df['match_minute'] == minute)]
    if latest.empty:
        print("⚠️ No data found for that minute. Try a different one.")
        return None

    latest_json = latest[['player_api_id', 'match_minute', 'physical_fatigue',
                          'mental_fatigue', 'combined_fatigue']].iloc[-1].to_dict()

    with open(path, 'w') as f:
        json.dump(latest_json, f, indent=2)
    print(f"✅ Exported prediction JSON → {path}")
    return latest_json
