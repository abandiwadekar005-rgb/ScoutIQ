# ScoutIQ

A football scouting tool that finds statistically similar, cheaper alternatives to any player, using unsupervised machine learning on per-90-minute performance data.

# The problem

Recruitment tools like Transfermarkt let you filter players by position, league, and market value, but they don't compare players on how they actually play — their statistical playing style, how effectively they execute it per 90 minutes, and whether a cheaper player exists with a near-identical profile. ScoutIQ addresses that gap directly.

# How it works

1) Cluster players by playstyle. Within each position group (e.g. strikers, central midfielders, centre-backs), players are clustered using K-Means on their scaled per-90 statistics — separating, for example, poacher-style strikers from creative forwards, or ball-playing centre-backs from traditional stoppers.

2) Choose K statistically, not arbitrarily. The number of clusters per position is selected by testing a range of K values and picking the one with the highest silhouette score, rather than guessing a fixed number.

3) Rank by similarity within the cluster. Given a target player, other members of their cluster are ranked by Euclidean distance on their scaled feature vectors — closer distance means a more statistically similar playing style.

4) Filter by price. Results are filtered to only players cheaper than the target, surfacing genuine "similar but more affordable" alternatives.

5) Present it simply. A Streamlit interface lets a user search a player, adjust a price ceiling, and see ranked alternatives with the underlying stats visible — so the recommendation isn't a black box.

# Tech stack
- Python — core language
- pandas — data loading, cleaning, manipulation
- scikit-learn — StandardScaler, KMeans, silhouette_score
- NumPy — distance calculations
- SQLite — player data and cluster assignment storage
- Streamlit — web interface

# Pipeline
Each module has a single responsibility and only depends on the one before it in the pipeline: database → preprocessing → clustering → similarity → app.

# Setup
- bash
- python3 -m venv venv
- source venv/bin/activate   # Windows: venv\Scripts\activate
- pip install pandas scikit-learn numpy streamlit

# Running it
bash
- python3 data/make_synthetic_data.py   # generate the dataset
- python3 database.py                    # load it into SQLite
- python3 clustering.py                  # cluster players, print silhouette scores
- streamlit run app.py                   # launch the web interface

Each module can also be run individually (python3 preprocessing.py, python3 similarity.py) to see its own sanity checks and test output.

# Current status

Built and verified against a small, hand-designed synthetic dataset — 21 players across three position groups (striker, central midfielder, centre-back), each assigned to a deliberate "archetype" (e.g. poacher vs. creative forward) so clustering correctness can be checked by hand. Three rows are deliberately invalid (below minutes threshold, missing stat, negative stat) to verify the validation logic.

## Verified results on this dataset:

Validation correctly drops all 3 invalid rows (21 → 18 players)
Clustering correctly separates every position into its two designed archetypes, with silhouette scores between 0.78 and 0.90
Similarity ranking correctly identifies cluster-mates, filters by price, and handles edge cases (no cheaper alternative exists, unknown player name)

## Not yet done:

1) Swap in a real dataset (FBref-style per-90 stats + Transfermarkt-style market values)

2) Expand beyond three position groups (fullback, winger, attacking midfielder, goalkeeper)

3) Deployment

4) Full test suite in tests/

## Design decisions worth noting

1) Scaling is done separately per position, not across the whole dataset — a striker's raw stat ranges are meaningless compared to a centre-back's, so scaling within each group is what makes the distance calculation meaningful.

2) K is chosen via silhouette score, not fixed, because the natural number of playstyle sub-groups genuinely differs by position.

3) Price filtering happens after ranking, not before — filtering first would risk excluding a statistically closer match purely on price before the more relevant comparison has been made.

4) Players below a minimum minutes-played threshold are excluded, since per-90 statistics for small sample sizes are unreliable.

# Limitations
1) Per-90 statistics can't capture tactical fit, off-ball movement, or intangible factors like leadership — this is a statistical similarity tool, not a complete recruitment judgement.

2) K-Means assumes roughly spherical, evenly-sized clusters, which may not hold for unconventional hybrid positional roles.

3) Currently limited to three position groups and a small dataset; results on a real, larger dataset have not yet been validated.
