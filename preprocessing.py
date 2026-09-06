"""M2 -- Preprocessing. Cleans, validates, and scales player data."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

from database import STAT_COLUMNS

MIN_MINUTES_THRESHOLD = 450


def clean_players(df):
    df = df.copy()
    df = df[df["minutes_played"] >= MIN_MINUTES_THRESHOLD]

    keep_mask = pd.Series(True, index=df.index)
    for position, cols in STAT_COLUMNS.items():
        pos_rows = df["position_group"] == position
        for col in cols:
            if col not in df.columns:
                continue
            null_here = pos_rows & df[col].isna()
            keep_mask &= ~null_here
            neg_here = pos_rows & (df[col] < 0)
            keep_mask &= ~neg_here

    return df[keep_mask].reset_index(drop=True)


def scale_by_position(df):
    results = {}
    for position, cols in STAT_COLUMNS.items():
        pos_df = df[df["position_group"] == position]
        if pos_df.empty:
            continue
        cols_present = [c for c in cols if c in pos_df.columns]
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(pos_df[cols_present])
        scaled_df = pd.DataFrame(scaled_values, columns=cols_present, index=pos_df.index)
        scaled_df.insert(0, "player_id", pos_df["player_id"].values)
        results[position] = (scaled_df, scaler)
    return results


if __name__ == "__main__":
    from database import get_all_players
    raw = get_all_players()
    cleaned = clean_players(raw)
    print(f"Raw: {len(raw)}, Cleaned: {len(cleaned)}")
