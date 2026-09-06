"""M1 -- Data Layer. Wraps all SQLite access for the players database."""
import sqlite3
import pandas as pd

DB_PATH = "scoutiq.db"

STAT_COLUMNS = {
    "ST": ["npxg_p90", "shots_p90", "touches_box_p90", "goals_p90", "sca_p90"],
    "CM": ["prog_passes_p90", "passes_final_third_p90", "tackles_int_p90",
           "pass_completion_pct", "touches_p90"],
    "CB": ["tackles_p90", "interceptions_p90", "clearances_p90",
           "aerial_win_pct", "prog_passes_p90"],
}


def create_tables(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cluster_assignments (
            player_id INTEGER,
            position_group TEXT,
            cluster_id INTEGER
        )
    """)
    conn.commit()
    conn.close()


def load_csv_into_db(csv_path, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_csv(csv_path)
    df.to_sql("players", conn, if_exists="replace", index=False)
    conn.close()
    return len(df)


def get_all_players(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM players", conn)
    conn.close()
    return df


def get_player_by_name(name, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM players WHERE name = ?", conn, params=(name,))
    conn.close()
    return None if df.empty else df.iloc[0]


def save_cluster_assignments(assignments_df, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM cluster_assignments")
    assignments_df[["player_id", "position_group", "cluster_id"]].to_sql(
        "cluster_assignments", conn, if_exists="append", index=False
    )
    conn.commit()
    conn.close()


def get_cluster_id(player_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT cluster_id FROM cluster_assignments WHERE player_id = ?",
        conn, params=(player_id,)
    )
    conn.close()
    return None if df.empty else int(df.iloc[0]["cluster_id"])


def get_cluster_members(position_group, cluster_id, exclude_player_id=None, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    query = """
        SELECT p.* FROM players p
        JOIN cluster_assignments c ON p.player_id = c.player_id
        WHERE c.position_group = ? AND c.cluster_id = ?
    """
    params = [position_group, cluster_id]
    if exclude_player_id is not None:
        query += " AND p.player_id != ?"
        params.append(exclude_player_id)
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df


if __name__ == "__main__":
    create_tables()
    n = load_csv_into_db("data/synthetic_players.csv")
    print(f"Loaded {n} players into {DB_PATH}")
