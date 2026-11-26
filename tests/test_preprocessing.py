"""
ScoutIQ development tests for M2 (preprocessing.py) - write-up sections
3.3.3 (cleaning / validation) and 3.3.4 (scaling).

Written with AI assistance (Claude Code). Tests designed from my NEA
write-up section 3.3; run and checked by me.

Prerequisites (run from the project root, in this order):
    1. data/synthetic_players.csv exists
    2. python database.py
    3. python clustering.py
Then:
    python tests/test_preprocessing.py

Plain Python script (not pytest). Each test prints one aligned line:
test ID, description, expected, actual, PASS/FAIL.
Expected values are the CORRECT behaviour, so a test can FAIL on purpose
where it exposes a known bug (currently 3.3.3-T1.9).

These tests only change in-memory copies of the player table, never
scoutiq.db itself.
"""
import os
import sys

import numpy as np
import pandas as pd

# Let this file import the project modules from the folder above tests/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from database import STAT_COLUMNS, get_all_players
from preprocessing import clean_players, scale_by_position
from clustering import PositionClusterer
from sklearn.metrics import silhouette_score

passed = 0
failed = 0


def check(test_id, description, actual, expected):
    """Compare a test result with its expected value and print one line.
    If 'actual' is a function it is called here, and any exception it
    raises is recorded as the actual value instead of crashing the file."""
    global passed, failed
    if callable(actual):
        try:
            actual = actual()
        except Exception as e:
            actual = f"{type(e).__name__}: {e}"
    if actual == expected:
        result = "PASS"
        passed += 1
    else:
        result = "FAIL"
        failed += 1
    exp_text = repr(expected)
    act_text = repr(actual)
    if len(act_text) > 90:
        act_text = act_text[:87] + "..."
    print(f"{test_id:<12} {description:<46} expected={exp_text:<30} actual={act_text:<30} {result}")


raw = get_all_players()


def raw_with_change(name, column, value):
    """Return a copy of the raw players with one value changed."""
    df = raw.copy()
    df.loc[df["name"] == name, column] = value
    return df


def kept(df, name):
    """True if the named player survives clean_players(df)."""
    return name in set(clean_players(df)["name"])


print("ScoutIQ - 3.3.3 cleaning tests (round T1)\n")

check("3.3.3-T1.1", "Clean all 21 players -> rows kept",
      lambda: len(clean_players(raw)), 18)
check("3.3.3-T1.2", "Poacher A (2400 min) kept",
      lambda: kept(raw, "Poacher A"), True)
check("3.3.3-T1.3", "Low Minutes Player removed",
      lambda: kept(raw, "Low Minutes Player"), False)
check("3.3.3-T1.4", "Null Stat Player removed",
      lambda: kept(raw, "Null Stat Player"), False)
check("3.3.3-T1.5", "Negative Stat Player removed",
      lambda: kept(raw, "Negative Stat Player"), False)
check("3.3.3-T1.6", "Poacher A at 450 minutes -> kept",
      lambda: kept(raw_with_change("Poacher A", "minutes_played", 450), "Poacher A"), True)
check("3.3.3-T1.7", "Poacher A at 449 minutes -> removed",
      lambda: kept(raw_with_change("Poacher A", "minutes_played", 449), "Poacher A"), False)
check("3.3.3-T1.8", "Poacher A goals_p90 = 0 -> kept",
      lambda: kept(raw_with_change("Poacher A", "goals_p90", 0), "Poacher A"), True)
check("3.3.3-T1.9", "Playmaker A pass_completion_pct 140 -> removed",
      lambda: kept(raw_with_change("Playmaker A", "pass_completion_pct", 140), "Playmaker A"), False)


def t3_1_10():
    """Cluster every position of the cleaned data (in memory, no DB write)
    and print each silhouette score."""
    cleaned = clean_players(raw)
    for position, (scaled_df, _) in scale_by_position(cleaned).items():
        features = scaled_df.drop(columns="player_id").values
        clusterer = PositionClusterer(position)
        labels = clusterer.fit(features)
        score = silhouette_score(features, labels)
        print(f"{'':<12} {position}: K={clusterer.best_k}, silhouette={score:.3f}")
    return "no crash"


check("3.3.3-T1.10", "Cluster cleaned data -> no crash", t3_1_10, "no crash")

print("\nScoutIQ - 3.3.4 scaling tests (round T1)\n")

cleaned = clean_players(raw)


def scaled_feature_tables():
    """{position: DataFrame of scaled stat columns only}"""
    return {pos: scaled_df.drop(columns="player_id")
            for pos, (scaled_df, _) in scale_by_position(cleaned).items()}


def t4_1_1():
    biggest = 0.0
    for table in scaled_feature_tables().values():
        biggest = max(biggest, table.mean().abs().max())
    return round(float(biggest), 6)


def t4_1_2():
    stds = set()
    for table in scaled_feature_tables().values():
        for col in table.columns:
            stds.add(round(float(np.std(table[col].values, ddof=0)), 6))
    return sorted(stds)


def distance_shares(table):
    """Each feature's share (%) of the mean squared pairwise distance."""
    values = table.values
    n = len(values)
    totals = np.zeros(values.shape[1])
    for i in range(n):
        for j in range(i + 1, n):
            totals += (values[i] - values[j]) ** 2
    return totals / totals.sum() * 100


def t4_1_3():
    shares = set()
    for table in scaled_feature_tables().values():
        for share in distance_shares(table):
            shares.add(round(float(share), 1))
    return sorted(shares)


def t4_1_4():
    no_cb = cleaned[cleaned["position_group"] != "CB"]
    return sorted(scale_by_position(no_cb).keys())


def t4_1_5():
    results = scale_by_position(cleaned)
    matches = []
    for position, (scaled_df, _) in results.items():
        input_ids = cleaned[cleaned["position_group"] == position]["player_id"].tolist()
        matches.append(scaled_df["player_id"].tolist() == input_ids
                       and scaled_df.columns[0] == "player_id")
    return all(matches)


check("3.3.4-T1.1", "Scaled mean ~0 (largest |mean|)", t4_1_1, 0.0)
check("3.3.4-T1.2", "Scaled std (ddof=0) ~1 (distinct values)", t4_1_2, [1.0])
check("3.3.4-T1.3", "Each feature's % share of mean sq. distance", t4_1_3, [20.0])
check("3.3.4-T1.4", "Empty CB group -> skipped, no crash", t4_1_4, ["CM", "ST"])
check("3.3.4-T1.5", "player_id kept beside values, input order", t4_1_5, True)

print(f"\n{passed} passed, {failed} failed")
