"""
M3 -- Clustering Engine. Groups players within each position into
statistically distinct playstyle clusters using K-Means, choosing K
via silhouette score rather than a fixed guess (Task 2, Section 3.2).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from database import save_cluster_assignments
from preprocessing import clean_players, scale_by_position


class PositionClusterer:
    """One instance per position group. Fits K-Means on that position's
    scaled features, choosing K by silhouette score over a small range."""

    def __init__(self, position_group, k_range=range(2, 6)):
        # k_range capped low (2-5) deliberately: our synthetic data only
        # has 2 archetypes per position and 9 players per position, so
        # testing up to K=10 (as in the original design) would produce
        # tiny, meaningless clusters on this dataset size. Real data
        # with more players per position can widen this back to 2-10.
        self.position_group = position_group
        self.k_range = k_range
        self.model = None
        self.best_k = None
        self.best_score = -1

    def fit(self, scaled_features):
        """scaled_features: numpy array, rows=players, cols=stats (no player_id)."""
        n_samples = scaled_features.shape[0]

        for k in self.k_range:
            if k >= n_samples:
                continue  # can't have more clusters than players
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(scaled_features)
            score = silhouette_score(scaled_features, labels)
            if score > self.best_score:
                self.best_score = score
                self.best_k = k

        self.model = KMeans(n_clusters=self.best_k, random_state=42, n_init=10)
        self.labels_ = self.model.fit_predict(scaled_features)
        return self.labels_


def cluster_all_positions(cleaned_df):
    """
    Runs PositionClusterer for every position group, saves the results
    to the cluster_assignments table, and returns a summary dict of
    {position: (best_k, best_score)} for reporting/evidence.
    """
    scaled_by_position = scale_by_position(cleaned_df)
    summary = {}
    all_assignments = []

    for position, (scaled_df, scaler) in scaled_by_position.items():
        feature_cols = [c for c in scaled_df.columns if c != "player_id"]
        features = scaled_df[feature_cols].values

        clusterer = PositionClusterer(position)
        labels = clusterer.fit(features)

        summary[position] = (clusterer.best_k, round(clusterer.best_score, 3))

        assignments = pd.DataFrame({
            "player_id": scaled_df["player_id"].values,
            "position_group": position,
            "cluster_id": labels,
        })
        all_assignments.append(assignments)

    combined = pd.concat(all_assignments, ignore_index=True)
    save_cluster_assignments(combined)
    return summary


if __name__ == "__main__":
    from database import get_all_players

    raw = get_all_players()
    cleaned = clean_players(raw)
    print(f"Clustering {len(cleaned)} cleaned players...\n")

    summary = cluster_all_positions(cleaned)

    print("Clustering summary (position: best_k, silhouette_score):")
    for position, (k, score) in summary.items():
        print(f"  {position}: K={k}, silhouette={score}")

    # Manual sanity check: print which cluster each player landed in,
    # so we can eyeball whether Poachers/Creators (etc.) actually split
    # into separate clusters as designed.
    import sqlite3
    conn = sqlite3.connect("scoutiq.db")
    check = pd.read_sql("""
        SELECT p.name, p.position_group, c.cluster_id
        FROM players p JOIN cluster_assignments c ON p.player_id = c.player_id
        ORDER BY p.position_group, c.cluster_id
    """, conn)
    conn.close()
    print("\nCluster assignments:")
    print(check.to_string(index=False))
