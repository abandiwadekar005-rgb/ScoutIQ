# ScoutIQ — Project Structure Guide

## How to use this guide

Files marked **[DONE]** are already built and verified — read them as
worked examples of the pattern (docstrings, function shape, how they
connect to database.py). Files marked **[YOUR TURN]** are skeletons:
docstrings, function signatures, and comments explain exactly what
each piece needs to do and why, but the logic inside is for you to
write. This is deliberate — for the NEA, and for interviews, you need
to be able to explain and defend every line, which only happens if
you actually write it.

Build order matches the M2→M5 dependency chain from the project plan.
Each file only needs the one(s) before it to already work.

```
scoutiq/
├── data/
│   ├── make_synthetic_data.py   [DONE] — generates data/synthetic_players.csv
│   └── synthetic_players.csv    [DONE] — 21 rows, 18 valid + 3 deliberately broken
│
├── database.py                  [DONE] — M1: SQLite access layer
├── preprocessing.py             [DONE] — M2a: cleaning + validation + scaling
├── clustering.py                [YOUR TURN] — M2b: K-Means + silhouette scoring
├── similarity.py                [YOUR TURN] — M3: distance ranking + price filter
├── app.py                       [YOUR TURN] — M4: Streamlit interface
│
└── tests/
    ├── test_preprocessing.py    [YOUR TURN] — unit tests for preprocessing.py
    ├── test_clustering.py       [YOUR TURN] — unit tests for clustering.py
    └── test_similarity.py       [YOUR TURN] — unit tests for similarity.py
```

## The data flow (why the files are ordered this way)

```
synthetic_players.csv
        │
        ▼
   database.py          (loads CSV → SQLite "players" table)
        │
        ▼
 preprocessing.py        (players table → cleaned, per-position scaled arrays)
        │
        ▼
  clustering.py           (scaled arrays → cluster_assignments table, via database.py)
        │
        ▼
  similarity.py             (target player + cluster_assignments → ranked, filtered list)
        │
        ▼
     app.py                  (user's search → calls similarity.py → displays results)
```

Each arrow is a function call, not a copy-paste — e.g. `clustering.py`
imports `scale_by_position` from `preprocessing.py` rather than
re-doing scaling itself. Keeping each file to one responsibility is
what Task 2's decomposition section argues for, and it's also just
easier to test and debug one piece at a time.

## What to do next

1. Open `clustering.py`, read the docstrings and TODOs, write the logic.
2. Run it directly (`python3 clustering.py`) — it has a manual sanity
   check built into its `if __name__ == "__main__"` block, same
   pattern as `preprocessing.py`.
3. Once clustering works, move to `similarity.py`, then `app.py`.
4. Tests can be written alongside each file, or afterward — whichever
   keeps you moving. Don't let "write tests first" block progress if
   it's slowing you down; get the pipeline working end-to-end first,
   then backfill rigorous tests for your Task 3 evidence.
