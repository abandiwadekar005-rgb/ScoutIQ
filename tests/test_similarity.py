"""
ScoutIQ development tests for M4 (similarity.py) - write-up sections
3.3.8 (Euclidean distance ranking), 3.3.9 (price filter) and 3.3.11
(error handling).

Written with AI assistance (Claude Code). Tests designed from my NEA
write-up section 3.3; run and checked by me.

Prerequisites (run from the project root, in this order):
    1. data/synthetic_players.csv exists
    2. python database.py
    3. python clustering.py
Then:
    python tests/test_similarity.py

Plain Python script (not pytest). Each test prints one aligned line:
test ID, description, expected, actual, PASS/FAIL.
Expected values are the CORRECT behaviour, so a test can FAIL on purpose
where it exposes a known bug (currently 3.3.8-T1.7 and 3.3.11-T1.10).

Tests that change data run on a temporary copy of scoutiq.db (DB_PATH is
the relative path "scoutiq.db", so chdir-ing into a temp folder is enough).
The missing-database test chdirs into an empty temp folder.
"""
import os
import shutil
import sqlite3
import sys
import tempfile
import time

import numpy as np

# Let this file import the project modules from the folder above tests/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from database import DB_PATH
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
    print(f"{test_id:<13} {description:<46} expected={exp_text:<30} actual={act_text:<30} {result}")


def run_in_temp_copy(test_function, sql_changes=()):
    """Copy scoutiq.db into a temporary folder, chdir there, apply any SQL
    changes, run test_function, then chdir back."""
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


def result_names(name):
    """Names returned by find_alternatives, in order (or the error text)."""
    results, error = find_alternatives(name)
    if error:
        return error
    if results.empty:
        return []
    return results["name"].tolist()


def row_count(name):
    results, error = find_alternatives(name)
    return error if error else len(results)


def error_of(name):
    return find_alternatives(name)[1]


print("ScoutIQ - 3.3.8 similarity tests (round T1)\n")


def t8_1_4():
    results, error = find_alternatives("Poacher C")
    distances = results["distance"].tolist()
    return distances == sorted(distances) and len(distances) > 0


def t8_1_6():
    runs = 20
    start = time.perf_counter()
    for _ in range(runs):
        find_alternatives("Poacher C")
    average = (time.perf_counter() - start) / runs
    print(f"{'':<13} average over {runs} searches: {average * 1000:.1f} ms")
    return average < 1.0


def t8_1_7():
    def inside_copy():
        results, error = find_alternatives("Playmaker A")
        if error:
            return error
        return (results["name"].tolist(), [round(d, 3) for d in results["distance"]])
    return run_in_temp_copy(
        inside_copy,
        ["UPDATE players SET minutes_played = 100 WHERE name = 'Playmaker B'"])


check("3.3.8-T1.1", "norm between (0,0) and (3,4)",
      lambda: float(np.linalg.norm(np.array([0, 0]) - np.array([3, 4]))), 5.0)
check("3.3.8-T1.2", "Poacher A not in its own results",
      lambda: "Poacher A" in result_names("Poacher A"), False)
check("3.3.8-T1.3", "Poacher C -> only Poachers returned",
      lambda: sorted(result_names("Poacher C")), ["Poacher A", "Poacher B"])
check("3.3.8-T1.4", "Poacher C results in ascending distance", t8_1_4, True)
check("3.3.8-T1.5", "Creator A -> Creator B and C",
      lambda: sorted(result_names("Creator A")), ["Creator B", "Creator C"])
check("3.3.8-T1.6", "Poacher C x20 -> average under 1 second", t8_1_6, True)
check("3.3.8-T1.7", "Playmaker B 100 min, no recluster -> A", t8_1_7,
      (["Playmaker C"], [0.326]))

print("\nScoutIQ - 3.3.9 price filter tests (round T1)\n")

check("3.3.9-T1.1", "Poacher A (£40m) -> results",
      lambda: result_names("Poacher A"), ["Poacher B"])
check("3.3.9-T1.2", "Poacher C (£55m) -> results",
      lambda: sorted(result_names("Poacher C")), ["Poacher A", "Poacher B"])
check("3.3.9-T1.3", "Poacher B value = £40m, search Poacher A",
      lambda: run_in_temp_copy(
          lambda: row_count("Poacher A"),
          ["UPDATE players SET market_value = 40000000 WHERE name = 'Poacher B'"]), 0)
check("3.3.9-T1.4", "Destroyer B -> rows returned",
      lambda: row_count("Destroyer B"), 0)

print("\nScoutIQ - 3.3.11 error handling tests (round T1)\n")

NOT_CLUSTERED = "has not been clustered yet"


def contains(text, phrase):
    return text is not None and phrase in text


def t11_1_10():
    """No scoutiq.db at all: should return a clear error message."""
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            results, error = find_alternatives("Poacher A")
        finally:
            os.chdir(original_dir)
    return "clear error message" if results is None and error else (results, error)


check("3.3.11-T1.1", "'Nonexistent Player' -> message",
      lambda: error_of("Nonexistent Player"), "No player found named 'Nonexistent Player'")
check("3.3.11-T1.2", "'' (empty) -> not-found message",
      lambda: error_of(""), "No player found named ''")
check("3.3.11-T1.3", "Low Minutes Player -> not clustered msg",
      lambda: contains(error_of("Low Minutes Player"), NOT_CLUSTERED), True)
check("3.3.11-T1.4", "Null Stat Player -> not clustered msg",
      lambda: contains(error_of("Null Stat Player"), NOT_CLUSTERED), True)
check("3.3.11-T1.5", "Negative Stat Player -> not clustered msg",
      lambda: contains(error_of("Negative Stat Player"), NOT_CLUSTERED), True)
check("3.3.11-T1.6", "Playmaker B 100 min -> excluded msg",
      lambda: run_in_temp_copy(
          lambda: contains(error_of("Playmaker B"), "was excluded during preprocessing"),
          ["UPDATE players SET minutes_played = 100 WHERE name = 'Playmaker B'"]), True)
check("3.3.11-T1.10", "No scoutiq.db -> clear error, no crash",
      t11_1_10, "clear error message")

print(f"\n{passed} passed, {failed} failed")
