"""
ScoutIQ development tests for M1 (database.py) - write-up section 3.3.2.

Run from the project folder, after the data has been generated:
    python3 make_synthetic_data.py
    python3 database.py
    python3 clustering.py
    python3 tests/test_database.py

Each line printed matches a row of the 3.3.2 test table.
"""
import os
import sqlite3
import sys

# Let this file import database.py from the project folder above tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import (
    DB_PATH,
    get_cluster_id,
    get_cluster_members,
    get_player_by_name,
    load_csv_into_db,
)

CSV_PATH = "data/synthetic_players.csv"
passed = 0
failed = 0


def check(test_id, description, actual, expected):
    """Compare a test result with its expected value and print one line."""
    global passed, failed
    if actual == expected:
        result = "PASS"
        passed += 1
    else:
        result = "FAIL"
        failed += 1
    print(f"{test_id:<12} {description:<32} expected={expected!r:<28} actual={actual!r:<28} {result}")


def name_of(player):
    """Return a player's name, or None if no player was found."""
    return None if player is None else player["name"]


print("ScoutIQ - 3.3.2 database tests (round T1)\n")

# T1.1  Normal: the CSV loads every player
rows_loaded = load_csv_into_db(CSV_PATH)
check("3.3.2-T1.1", "Load CSV", rows_loaded, 21)

# T1.2  Normal: a valid player is found
check("3.3.2-T1.2", "Retrieve valid player",
      name_of(get_player_by_name("Poacher A")), "Poacher A")

# T1.3  Erroneous: a player who does not exist
check("3.3.2-T1.3", "Retrieve non-existent player",
      name_of(get_player_by_name("Nonexistent Player")), None)

# T1.4  Erroneous: an empty search
check("3.3.2-T1.4", "Empty search",
      name_of(get_player_by_name("")), None)

# T1.5  Boundary: the same name in lower case
check("3.3.2-T1.5", "Case of name ('poacher a')",
      name_of(get_player_by_name("poacher a")), "Poacher A")

# T1.6  Normal: both tables exist
conn = sqlite3.connect(DB_PATH)
tables = sorted(row[0] for row in conn.execute(
    "SELECT name FROM sqlite_master WHERE type = 'table'"))
conn.close()
check("3.3.2-T1.6", "Both tables exist", tables, ["cluster_assignments", "players"])

# T1.7  Normal: cluster-mates of Poacher A, excluding Poacher A.
# The cluster number is looked up, not assumed, because K-Means
# numbers its clusters arbitrarily.
target = get_player_by_name("Poacher A")
cluster = get_cluster_id(int(target["player_id"]))
members = get_cluster_members("ST", cluster, exclude_player_id=int(target["player_id"]))
check("3.3.2-T1.7", "Cluster members, excl. target",
      sorted(members["name"]), ["Poacher B", "Poacher C"])

print(f"\n{passed} passed, {failed} failed")