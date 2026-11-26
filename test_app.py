"""
ScoutIQ development tests for M5 (app.py, the Streamlit interface) -
write-up sections 3.3.9 (price slider), 3.3.10 (interface), 3.3.11
(error / empty-state messages) and 3.3.12 (end-to-end searches).

Written with AI assistance (Claude Code). Tests designed from my NEA
write-up section 3.3; run and checked by me.

Prerequisites (run from the project root, in this order):
    1. data/synthetic_players.csv exists
    2. python database.py
    3. python clustering.py
Then:
    python tests/test_app.py

Uses streamlit.testing.v1.AppTest to run app.py without a browser.
Plain Python script (not pytest). Each test prints one aligned line:
test ID, description, expected, actual, PASS/FAIL.
3.3.10-T1.8 is a manual stakeholder test and is only printed here.
These tests only read scoutiq.db; nothing is changed.
"""
import logging
import os
import re
import sys

# Let this file import the project modules from the folder above tests/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from streamlit import logger as st_logger
from streamlit.testing.v1 import AppTest

# Hide Streamlit's harmless "missing ScriptRunContext" warnings in bare mode
st_logger.set_log_level("error")
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").disabled = True

APP_PATH = os.path.join(ROOT, "app.py")
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
    print(f"{test_id:<13} {description:<44} expected={exp_text:<30} actual={act_text:<30} {result}")


def run_app(player=None, max_price=None):
    """Run app.py, optionally choose a player and move the slider.
    Raises if the app itself raised an exception."""
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    if player is not None:
        at.selectbox[0].select(player).run()
    if max_price is not None:
        at.slider[0].set_value(max_price).run()
    if at.exception:
        raise RuntimeError(at.exception[0].message)
    return at


def expander_names(at):
    """Player names at the start of each expander label, in display order."""
    return [e.label.split(" — ")[0] for e in at.expander]


def expander_distances(at):
    return [float(re.search(r"similarity distance: ([\d.]+)", e.label).group(1))
            for e in at.expander]


def info_messages(at):
    return [i.value for i in at.info]


def has_info(at, phrase):
    return any(phrase in message for message in info_messages(at))


def slider_range(at):
    return (int(at.slider[0].min), int(at.slider[0].max))


print("ScoutIQ - 3.3.9 price slider tests (round T1)\n")

check("3.3.9-T1.5", "Poacher C, slider £20m -> shown",
      lambda: expander_names(run_app("Poacher C", 20_000_000)), ["Poacher B"])
check("3.3.9-T1.6", "Poacher C, slider £5m -> info message",
      lambda: has_info(run_app("Poacher C", 5_000_000), "No alternatives within that price range"), True)


def t9_1_7():
    distances = expander_distances(run_app("Poacher C"))
    return distances == sorted(distances) and len(distances) > 0


check("3.3.9-T1.7", "Poacher C -> expanders ascending distance", t9_1_7, True)
check("3.3.9-T1.8", "Destroyer B -> slider (min, max)",
      lambda: slider_range(run_app("Destroyer B")), (0, 10_000_000))

print("\nScoutIQ - 3.3.10 interface tests (round T1)\n")


def t10_1_2():
    options = list(run_app().selectbox[0].options)
    return (len(options), options == sorted(options))


def t10_1_4():
    for m in run_app("Poacher C").markdown:
        if "minutes played" in m.value:
            return m.value.replace("**", "")
    return "no summary line found"


def t10_1_7():
    first = run_app("Poacher C").expander[0]
    has_heading = any("Why this player?" in m.value for m in first.markdown)
    return has_heading and len(first.table) == 1


check("3.3.10-T1.1", "No selection -> prompt text",
      lambda: [m.value for m in run_app().markdown], ["Select a player above to get started."])
check("3.3.10-T1.2", "Selectbox (option count, alphabetical)", t10_1_2, (18, True))
check("3.3.10-T1.3", "'Low Minutes Player' in options",
      lambda: "Low Minutes Player" in run_app().selectbox[0].options, False)
check("3.3.10-T1.4", "Poacher C -> summary line", t10_1_4,
      "Poacher C — ST — £55,000,000 — 2600 minutes played")
check("3.3.10-T1.5", "Poacher C -> expanders in order",
      lambda: expander_names(run_app("Poacher C")), ["Poacher A", "Poacher B"])
check("3.3.10-T1.6", "Playmaker A -> number of expanders",
      lambda: len(run_app("Playmaker A").expander), 2)
check("3.3.10-T1.7", "Expander has 'Why this player?' table", t10_1_7, True)
print(f"{'3.3.10-T1.8':<13} MANUAL - stakeholder usability test, recorded separately")

print("\nScoutIQ - 3.3.11 interface messages tests (round T1)\n")

check("3.3.11-T1.7", "Destroyer B -> 'no cheaper' info",
      lambda: has_info(run_app("Destroyer B"), "No cheaper statistically similar players found"), True)
check("3.3.11-T1.8", "Poacher C, £5m -> raise-the-slider info",
      lambda: has_info(run_app("Poacher C", 5_000_000), "Try raising the slider"), True)
check("3.3.11-T1.9", "Destroyer B -> slider limited (min, max)",
      lambda: slider_range(run_app("Destroyer B")), (0, 10_000_000))

print("\nScoutIQ - 3.3.12 end-to-end tests (round T1)\n")


def t12_1_5():
    at = run_app("Destroyer B")
    return (len(at.expander), len(info_messages(at)))


def t12_1_6():
    first = [e.label for e in run_app("Poacher C").expander]
    second = [e.label for e in run_app("Poacher C").expander]
    return first == second and len(first) > 0


check("3.3.12-T1.1", "Poacher C full budget -> order",
      lambda: expander_names(run_app("Poacher C")), ["Poacher A", "Poacher B"])
check("3.3.12-T1.2", "Playmaker A -> results",
      lambda: sorted(expander_names(run_app("Playmaker A"))), ["Playmaker B", "Playmaker C"])
check("3.3.12-T1.3", "Sweeper A -> order",
      lambda: expander_names(run_app("Sweeper A")), ["Sweeper C", "Sweeper B"])
check("3.3.12-T1.4", "Poacher C, budget £20m -> results",
      lambda: expander_names(run_app("Poacher C", 20_000_000)), ["Poacher B"])
check("3.3.12-T1.5", "Destroyer B -> (expanders, info messages)", t12_1_5, (0, 1))
check("3.3.12-T1.6", "Same search twice -> identical labels", t12_1_6, True)

print(f"\n{passed} passed, {failed} failed")
