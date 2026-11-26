"""
ScoutIQ comparison / evidence runs for write-up sections 3.3.3, 3.3.4,
3.3.5, 3.3.7 and 3.3.9. These are NOT pass/fail tests: they print
labelled output showing what happens WITHOUT a design decision (no
cleaning, no scaling, wrong K, one shared table, no price filter) so it
can be compared with the real pipeline.

Written with AI assistance (Claude Code). Tests designed from my NEA
write-up section 3.3; run and checked by me.

Prerequisites (run from the project root, in this order):
    1. data/synthetic_players.csv exists
    2. python database.py
    3. python clustering.py
Then:
    python tests/comparison_runs.py

Nothing is written to scoutiq.db.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

# Let this file import the project modules from the folder above tests/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from database import STAT_COLUMNS, get_all_players, get_cluster_id, get_cluster_members, get_player_by_name
from preprocessing import clean_players, scale_by_position
from similarity import _get_scaled_features_lookup

pd.set_option("display.width", 200)
raw = get_all_players()
cleaned = clean_players(raw)


def heading(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def distance_shares(values):
    """Each feature's share (%) of the mean squared pairwise distance."""
    n = len(values)
    totals = np.zeros(values.shape[1])
    for i in range(n):
        for j in range(i + 1, n):
            totals += (values[i] - values[j]) ** 2
    return totals / totals.sum() * 100


# ---------------------------------------------------------------------------
heading("3.3.3 - Clustering WITHOUT cleaning (invalid rows left in)")

print("\n[CM] K-Means on raw CM data including Null Stat Player:")
raw_cm = raw[raw["position_group"] == "CM"]
try:
    KMeans(n_clusters=2, random_state=42, n_init=10).fit(raw_cm[STAT_COLUMNS["CM"]].values)
    print("  No error raised")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

for position, invalid in [("ST", "Low Minutes Player"), ("CB", "Negative Stat Player")]:
    print(f"\n[{position}] uncleaned data including {invalid} "
          f"(scaled per position, as the pipeline would):")
    pos_df = raw[raw["position_group"] == position]
    features = StandardScaler().fit_transform(pos_df[STAT_COLUMNS[position]])
    for k in range(2, 6):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(features)
        print(f"  K={k}: silhouette={silhouette_score(features, labels):.3f}")
    labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(features)
    table = pd.DataFrame({"name": pos_df["name"].values, "cluster (K=2)": labels})
    print(table.sort_values("cluster (K=2)").to_string(index=False))

print("\nFor comparison, cleaned data silhouette at K=2:")
for position, (scaled_df, _) in scale_by_position(cleaned).items():
    features = scaled_df.drop(columns="player_id").values
    labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(features)
    print(f"  {position}: {silhouette_score(features, labels):.3f}")

# ---------------------------------------------------------------------------
heading("3.3.4 - Each feature's share of distance, UNSCALED vs SCALED")

scaled = scale_by_position(cleaned)
for position, cols in STAT_COLUMNS.items():
    pos_df = cleaned[cleaned["position_group"] == position]
    unscaled_share = distance_shares(pos_df[cols].values.astype(float))
    scaled_share = distance_shares(scaled[position][0][cols].values)
    table = pd.DataFrame({"feature": cols,
                          "unscaled %": unscaled_share.round(1),
                          "scaled %": scaled_share.round(1)})
    print(f"\n[{position}]")
    print(table.to_string(index=False))

print("\nPlaymaker A's distance to every CM player, unscaled vs scaled:")
cm_df = cleaned[cleaned["position_group"] == "CM"].reset_index(drop=True)
cm_scaled = scaled["CM"][0].reset_index(drop=True)
cols = STAT_COLUMNS["CM"]
a_index = cm_df.index[cm_df["name"] == "Playmaker A"][0]
rows = []
for i, row in cm_df.iterrows():
    if i == a_index:
        continue
    rows.append({
        "player": row["name"],
        "unscaled": np.linalg.norm(cm_df.loc[a_index, cols].values.astype(float)
                                   - row[cols].values.astype(float)),
        "scaled": np.linalg.norm(cm_scaled.loc[a_index, cols].values - cm_scaled.loc[i, cols].values),
    })
table = pd.DataFrame(rows)
table["unscaled rank"] = table["unscaled"].rank().astype(int)
table["scaled rank"] = table["scaled"].rank().astype(int)
print(table.round(3).sort_values("scaled").to_string(index=False))

# ---------------------------------------------------------------------------
heading("3.3.5 - ST labels at K=2 vs K=4")

st_scaled, _ = scaled["ST"]
st_features = st_scaled.drop(columns="player_id").values
st_names = [raw[raw["player_id"] == pid]["name"].iloc[0] for pid in st_scaled["player_id"]]
table = pd.DataFrame({"name": st_names})
for k in (2, 4):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(st_features)
    table[f"K={k}"] = labels
    print(f"K={k}: silhouette={silhouette_score(st_features, labels):.3f}")
print(table.to_string(index=False))

# ---------------------------------------------------------------------------
heading("3.3.7 - Missing values if all positions shared ONE set of columns")

all_cols = []
for cols in STAT_COLUMNS.values():
    for col in cols:
        if col not in all_cols:
            all_cols.append(col)
print(f"Distinct stat columns across all positions: {len(all_cols)}")
print(f"Cleaned players: {len(cleaned)}\n")
missing = cleaned[all_cols].isna().sum()
print(pd.DataFrame({"column": all_cols,
                    "missing": missing.values,
                    "of": len(cleaned)}).to_string(index=False))
print(f"\nTotal missing cells: {int(missing.sum())} of {len(cleaned) * len(all_cols)}")

# ---------------------------------------------------------------------------
heading("3.3.9 - Poacher A's cluster-mates ranked BEFORE the price filter")

target = get_player_by_name("Poacher A")
target_id = int(target["player_id"])
lookup = _get_scaled_features_lookup(cleaned)
mates = get_cluster_members("ST", get_cluster_id(target_id), exclude_player_id=target_id)
mates["distance"] = [np.linalg.norm(lookup[target_id]["features"] - lookup[int(pid)]["features"])
                     for pid in mates["player_id"]]
mates = mates.sort_values("distance")
mates["cheaper than £{:,}?".format(int(target["market_value"]))] = (
    mates["market_value"] < target["market_value"])
print(mates[["name", "market_value", "distance",
             mates.columns[-1]]].round(3).to_string(index=False))
