"""
M5 -- Interface Layer. Streamlit app: search a player, filter by
price, see ranked statistically-similar cheaper alternatives.
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd

from database import get_all_players, DB_PATH, STAT_COLUMNS
from preprocessing import clean_players
from similarity import find_alternatives

st.set_page_config(page_title="ScoutIQ", layout="centered")
st.title("ScoutIQ")
st.caption("Find statistically similar, cheaper alternatives to any player.")

# Load and cache the player list + cleaned dataset once per session
# rather than re-querying SQLite on every interaction.
@st.cache_data
def load_players():
    raw = get_all_players()
    cleaned = clean_players(raw)
    return raw, cleaned

raw_players, cleaned_players = load_players()

player_names = sorted(cleaned_players["name"].unique())
selected_name = st.selectbox(
    "Search for a player",
    options=player_names,
    index=None,
    placeholder="Type a player name...",
)

if selected_name:
    target_row = raw_players[raw_players["name"] == selected_name].iloc[0]
    target_value = int(target_row["market_value"])

    st.write(f"**{selected_name}** — {target_row['position_group']} — "
             f"£{target_value:,} — {int(target_row['minutes_played'])} minutes played")

    max_price = st.slider(
        "Maximum market value for alternatives (£)",
        min_value=0,
        max_value=target_value,
        value=target_value,
        step=1_000_000,
        format="£%d",
    )

    results, error = find_alternatives(selected_name, cleaned_df=cleaned_players)

    if error:
        st.error(error)
    elif results.empty:
        # Usability feature from Task 2 Section 4: clear empty-state
        # message rather than a blank table, since a blank table with
        # no explanation would look like a broken app, not "no results".
        st.info(
            f"No cheaper statistically similar players found for {selected_name} "
            f"in the current dataset. This player may already be the cheapest "
            f"in their playstyle cluster."
        )
    else:
        filtered = results[results["market_value"] <= max_price]

        if filtered.empty:
            st.info("No alternatives within that price range. Try raising the slider.")
        else:
            st.subheader(f"Alternatives to {selected_name}")
            for _, row in filtered.iterrows():
                with st.expander(f"{row['name']} — £{int(row['market_value']):,} "
                                  f"(similarity distance: {row['distance']:.2f})"):
                    stat_cols = STAT_COLUMNS.get(target_row["position_group"], [])
                    stats_display = {col: row[col] for col in stat_cols if col in row}
                    st.write("**Why this player?** Key per-90 stats:")
                    st.table(pd.DataFrame([stats_display]))
else:
    st.write("Select a player above to get started.")
