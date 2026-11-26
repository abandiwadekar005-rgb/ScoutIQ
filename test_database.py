"""
ScoutIQ development tests for M1 (database.py) - write-up section 3.3.2.

Written with AI assistance (Claude Code). Tests designed from my NEA
write-up section 3.3; run and checked by me.

Prerequisites (run from the project root, in this order):
    1. data/synthetic_players.csv exists
    2. python database.py
    3. python clustering.py
Then:
    python tests/test_database.py

Plain Python script (not pytest). Each test prints one aligned line:
test ID, description, expected, actual, PASS/FAIL.

T1.1 reloads the CSV, which rewrites the players table, so it runs on a
temporary copy of scoutiq.db and never touches the real database.
"""
import os
import shutil
import sqlite3
import sys
import tempfile

# Let this file import the project modules from the folder above tests/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from database import (
    DB_PATH,
    get_cluster_id,
    get_cluster_members,
    get_player_by_name,
    load_csv_into_db,
)

CSV_PATH = os.path.join(ROOT, "data", "synthetic_players.csv")
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
    print(f"{test_id:<12} {description:<34} expected={exp_text:<30} actual={act_text:<30} {result}")


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


def name_of(player):
    """Return a player's name, or None if no player was found."""
    return None if player is None else player["name"]


print("ScoutIQ - 3.3.2 database tests (round T1)\n")

# T1.1  Normal: the CSV loads every player (on a temp copy of the DB)
check("3.3.2-T1.1", "Load CSV",
      lambda: run_in_temp_copy(lambda: load_csv_into_db(CSV_PATH)), 21)

# T1.2  Normal: a valid player is found
check("3.3.2-T1.2", "Retrieve valid player",
      lambda: name_of(get_player_by_name("Poacher A")), "Poacher A")

# T1.3  Erroneous: a player who does not exist
check("3.3.2-T1.3", "Retrieve non-existent player",
      lambda: name_of(get_player_by_name("Nonexistent Player")), None)

# T1.4  Erroneous: an empty search
check("3.3.2-T1.4", "Empty search",
      lambda: name_of(get_player_by_name("")), None)

# T1.5  Boundary: the same name in lower case
check("3.3.2-T1.5", "Case of name ('poacher a')",
      lambda: name_of(get_player_by_name("poacher a")), "Poacher A")


# T1.6  Normal: both tables exist
def t1_6():
    conn = sqlite3.connect(DB_PATH)
    tables = sorted(row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"))
    conn.close()
    return tables


check("3.3.2-T1.6", "Both tables exist", t1_6, ["cluster_assignments", "players"])


# T1.7  Normal: cluster-mates of Poacher A, excluding Poacher A.
# The cluster number is looked up, not assumed, because K-Means
# numbers its clusters arbitrarily.
def t1_7():
    target = get_player_by_name("Poacher A")
    cluster = get_cluster_id(int(target["player_id"]))
    members = get_cluster_members("ST", cluster, exclude_player_id=int(target["player_id"]))
    return sorted(members["name"])


check("3.3.2-T1.7", "Cluster members, excl. target", t1_7, ["Poacher B", "Poacher C"])

print(f"\n{passed} passed, {failed} failed")
