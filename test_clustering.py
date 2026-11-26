"""
ScoutIQ development tests for M3 (clustering.py) - write-up sections
3.3.5 (K-Means clustering), 3.3.6 (choosing K by silhouette score) and
3.3.7 (clusters stay within one position group).

Written with AI assistance (Claude Code). Tests designed from my NEA
write-up section 3.3; run and checked by me.

Prerequisites (run from the project root, in this order):
    1. data/synthetic_players.csv exists
    2. python database.py
    3. python clustering.py
Then:
    python tests/test_clustering.py

Plain Python script (not pytest). Each test prints one aligned line:
test ID, description, expected, actual, PASS/FAIL.
Expected values are the CORRECT behaviour, so a test can FAIL on purpose
where it exposes a known bug (currently 3.3.5-T1.5 and 3.3.6-T1.6).

Cluster numbers are always looked up with get_cluster_id(), because
K-Means numbers its clusters arbitrarily. 3.3.5-T1.1 rewrites the
cluster_assignments table, so it runs on a temporary copy of scoutiq.db.
"""
import os
import shutil
import sqlite3
import sys
import tempfile

import numpy as np
import pandas as pd

# Let this file import the project modules from the folder above tests/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from database import DB_PATH, get_all_players, get_cluster_id, get_cluster_members, get_player_by_name
from preprocessing import clean_players, scale_by_position
from clustering import PositionClusterer, cluster_all_positions
from similarity import find_alternatives

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


def run_in_temp_copy(test_function, sql_changes=()):
    """Copy scoutiq.db into a temporary folder, chdir there, apply any SQL
    changes, run test_function, then chdir back. The real DB is untouched
    because DB_PATH is the relative path 'scoutiq.db'."""
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(os.path.join(ROOT, DB_PATH), os.path.join(tmp, DB_PATH))
        os.chdir(tmp)
        try:
            conn = sqlite3.connect(DB_PATH)
            for sql in sql_changes:
                conn.execute(sql)
            conn.commit()
            conn.close()
            return test_function()
        finally:
            os.chdir(original_dir)


raw = get_all_players()
cleaned = clean_players(raw)
TRIOS = {"ST": ("Poacher", "Creator"), "CM": ("Playmaker", "Destroyer"), "CB": ("Sweeper", "Stopper")}


def player_id(name):
    return int(raw[raw["name"] == name]["player_id"].iloc[0])


def scaled_features(position):
    """(player_ids, scaled feature array) for one position of the cleaned data."""
    scaled_df, _ = scale_by_position(cleaned)[position]
    return scaled_df["player_id"].tolist(), scaled_df.drop(columns="player_id").values


def trios_correct(labels_by_name, position):
    """True if each archetype trio shares one label, and the two trios differ."""
    first, second = TRIOS[position]
    labels_a = {labels_by_name[f"{first} {x}"] for x in "ABC"}
    labels_b = {labels_by_name[f"{second} {x}"] for x in "ABC"}
    return len(labels_a) == 1 and len(labels_b) == 1 and labels_a != labels_b


def names_for(ids):
    return [raw[raw["player_id"] == pid]["name"].iloc[0] for pid in ids]


print("ScoutIQ - 3.3.5 clustering tests (round T1)\n")


def t5_1_1():
    def inside_copy():
        cluster_all_positions(cleaned)
        conn = sqlite3.connect(DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM cluster_assignments").fetchone()[0]
        conn.close()
        return count
    return run_in_temp_copy(inside_copy)


def t5_1_2():
    result = {}
    for position in TRIOS:
        names = [f"{t} {x}" for t in TRIOS[position] for x in "ABC"]
        labels = {name: get_cluster_id(player_id(name)) for name in names}
        result[position] = trios_correct(labels, position)
    return result


def t5_1_3():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT player_id FROM cluster_assignments WHERE player_id IN (19, 20, 21)").fetchall()
    conn.close()
    return [r[0] for r in rows]


def t5_1_4():
    _, features = scaled_features("ST")
    first = PositionClusterer("ST").fit(features).tolist()
    second = PositionClusterer("ST").fit(features).tolist()
    return first == second


def t5_1_5():
    ids, features = scaled_features("ST")
    labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(features)
    by_name = dict(zip(names_for(ids), labels))
    return len({by_name[f"Poacher {x}"] for x in "ABC"}) == 1


check("3.3.5-T1.1", "cluster_all_positions -> rows saved", t5_1_1, 18)
check("3.3.5-T1.2", "Archetype trios share one cluster each", t5_1_2,
      {"ST": True, "CM": True, "CB": True})
check("3.3.5-T1.3", "Player ids 19, 20, 21 not clustered", t5_1_3, [])
check("3.3.5-T1.4", "ST clustered twice -> identical labels", t5_1_4, True)
check("3.3.5-T1.5", "ST forced K=4 -> Poachers in one cluster", t5_1_5, True)

print("\nScoutIQ - 3.3.6 choosing K tests (round T1)\n")


def silhouette_scores():
    """{position: {k: score}} for K=2..5, same settings as PositionClusterer."""
    scores = {}
    for position in TRIOS:
        _, features = scaled_features(position)
        scores[position] = {}
        for k in range(2, 6):
            if k >= len(features):
                continue
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(features)
            scores[position][k] = silhouette_score(features, labels)
    return scores


def t6_1_1():
    scores = silhouette_scores()
    for position, by_k in scores.items():
        text = ", ".join(f"K={k}: {s:.3f}" for k, s in by_k.items())
        print(f"{'':<12} {position}: {text}")
    return sum(len(by_k) for by_k in scores.values())


def t6_1_2():
    result = {}
    for position in TRIOS:
        clusterer = PositionClusterer(position)
        clusterer.fit(scaled_features(position)[1])
        result[position] = clusterer.best_k
    return result


def t6_1_3():
    return all(-1 <= s <= 1 for by_k in silhouette_scores().values() for s in by_k.values())


def t6_1_4():
    result = {}
    for position in TRIOS:
        ids, features = scaled_features(position)
        labels = PositionClusterer(position).fit(features)
        result[position] = trios_correct(dict(zip(names_for(ids), labels)), position)
    return result


def t6_1_5():
    _, features = scaled_features("ST")
    PositionClusterer("ST").fit(features[:3])
    return "no crash"


def t6_1_6():
    _, features = scaled_features("ST")
    PositionClusterer("ST").fit(features[:2])
    return "handled, no crash"


check("3.3.6-T1.1", "Silhouette for K=2..5, 3 positions", t6_1_1, 12)
check("3.3.6-T1.2", "best_k for ST, CM, CB", t6_1_2, {"ST": 2, "CM": 2, "CB": 2})
check("3.3.6-T1.3", "Every score between -1 and 1", t6_1_3, True)
check("3.3.6-T1.4", "Labels at chosen K reproduce the trios", t6_1_4,
      {"ST": True, "CM": True, "CB": True})
check("3.3.6-T1.5", "3-player group -> no crash", t6_1_5, "no crash")
check("3.3.6-T1.6", "2-player group -> handled, no crash", t6_1_6, "handled, no crash")

print("\nScoutIQ - 3.3.7 position separation tests (round T1)\n")


def result_positions(name):
    """Sorted distinct positions in find_alternatives(name)'s results."""
    results, error = find_alternatives(name)
    if error:
        return error
    return sorted(results["position_group"].unique())


def t7_1_5():
    st_ids = set(get_cluster_members("ST", 0)["player_id"])
    cm_ids = set(get_cluster_members("CM", 0)["player_id"])
    return sorted(st_ids & cm_ids)


check("3.3.7-T1.1", "find_alternatives('Poacher C') -> positions",
      lambda: result_positions("Poacher C"), ["ST"])
check("3.3.7-T1.2", "find_alternatives('Playmaker A') -> positions",
      lambda: result_positions("Playmaker A"), ["CM"])
check("3.3.7-T1.3", "find_alternatives('Sweeper A') -> positions",
      lambda: result_positions("Sweeper A"), ["CB"])
check("3.3.7-T1.4", "find_alternatives('Stopper A') -> positions",
      lambda: result_positions("Stopper A"), ["CB"])
check("3.3.7-T1.5", "ST cluster 0 and CM cluster 0 shared ids", t7_1_5, [])

print(f"\n{passed} passed, {failed} failed")
