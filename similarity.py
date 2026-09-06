"""
M4 -- Similarity & Filter Engine. Given a target player, finds other
members of their cluster, ranks them by Euclidean distance on scaled
features, and filters to those cheaper than the target.
"""
import numpy as np
import pandas as pd
import sqlite3

from database import DB_PATH, STAT_COLUMNS, get_cluster_id, get_cluster_members, get_player_by_name
from preprocessing import scale_by_position, clean_players


def _get_scaled_features_lookup(cleaned_df):
    """Returns {player_id: {position_group, feature_array}} for every
    cleaned player, so distance can be computed without rescaling on
    every single query."""
    scaled_by_position = scale_by_position(cleaned_df)
    lookup = {}
    for position, (scaled_df, scaler) in scaled_by_position.items():
        feature_cols = [c for c in scaled_df.columns if c != "player_id"]
        for _, row in scaled_df.iterrows():
            lookup[int(row["player_id"])] = {
                "position_group": position,
                "features": row[feature_cols].values.astype(float),
            }
    return lookup


def find_alternatives(target_name, max_results=10, cleaned_df=None):
    """
    Returns a DataFrame of cheaper, statistically similar players to
    the named target, ranked by ascending distance (most similar first).
    """
    target = get_player_by_name(target_name)
    if target is None:
        return None, f"No player found named '{target_name}'"

    target_id = int(target["player_id"])
    position = target["position_group"]
    cluster_id = get_cluster_id(target_id)
    if cluster_id is None:
        return None, f"'{target_name}' has not been clustered yet (may have failed validation)."

    # Need scaled features for both the target and its cluster-mates.
    # Recompute the scaling lookup from the full cleaned dataset so the
    # target's own scaled position matches exactly what clustering used.
    if cleaned_df is None:
        from database import get_all_players
        cleaned_df = clean_players(get_all_players())
    lookup = _get_scaled_features_lookup(cleaned_df)

    if target_id not in lookup:
        return None, f"'{target_name}' was excluded during preprocessing (e.g. below minutes threshold)."

    target_features = lookup[target_id]["features"]

    candidates = get_cluster_members(position, cluster_id, exclude_player_id=target_id)
    if candidates.empty:
        return pd.DataFrame(), None  # valid outcome: no cluster-mates at all

    distances = []
    for _, row in candidates.iterrows():
        pid = int(row["player_id"])
        if pid not in lookup:
            continue
        dist = np.linalg.norm(target_features - lookup[pid]["features"])
        distances.append(dist)

    candidates = candidates.iloc[:len(distances)].copy()
    candidates["distance"] = distances
    candidates = candidates.sort_values("distance")

    # Price filter applied AFTER ranking, not before -- see Task 2,
    # Section 3.3 justification: ranking first preserves the integrity
    # of the similarity ordering before price is applied as a constraint.
    cheaper = candidates[candidates["market_value"] < target["market_value"]]

    return cheaper.head(max_results).reset_index(drop=True), None


if __name__ == "__main__":
    # Sanity checks matching Task 2's test plan (Section 6.1 / 6.2)

    print("--- Test 1: known similar players (Poacher A -> expect Poacher B, C) ---")
    result, error = find_alternatives("Poacher A")
    if error:
        print("ERROR:", error)
    else:
        print(result[["name", "market_value", "distance"]].to_string(index=False))

    print("\n--- Test 2: most expensive player in a cluster (Poacher C, 55m) ---")
    print("Expect: only cheaper Poachers returned, Creators excluded (different cluster)")
    result, error = find_alternatives("Poacher C")
    print(result[["name", "market_value", "distance"]].to_string(index=False) if not error else error)

    print("\n--- Test 3: cheapest player in their cluster (Destroyer B, 10m) ---")
    print("Expect: empty result -- no one in the cluster is cheaper")
    result, error = find_alternatives("Destroyer B")
    if error:
        print("ERROR:", error)
    else:
        print("Rows returned:", len(result))
        print(result.to_string(index=False) if not result.empty else "(empty, as expected)")

    print("\n--- Test 4: unknown player name ---")
    result, error = find_alternatives("Nonexistent Player")
    print("Error message:", error)
