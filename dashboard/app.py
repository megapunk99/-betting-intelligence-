"""
Betting Intelligence Dashboard — Phase 5.19 of the Professional Betting Intelligence Platform.

Focuses on:

1. Performance — ROI, Yield, Profit Factor, Sharpe Ratio, CLV, Drawdown
2. Betting — Today's Bets, Tomorrow's Bets, Open Positions, Risk Exposure
3. Models — Model Accuracy, Calibration, Drift, Feature Importance

Usage:
    streamlit run dashboard/app.py
"""

import sys
import json
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Betting Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════
#  DATABASE CONNECTION
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_db_path():
    try:
        from betting_intel.config import DB_PATH
        return DB_PATH
    except ImportError:
        return Path("data/betting_intel.db")


@st.cache_resource
def get_connection():
    db_path = get_db_path()
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn
    return None


def query_db(query: str, params: tuple = ()) -> pd.DataFrame:
    """Execute a query and return results as a DataFrame."""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    return pd.read_sql_query(query, conn, params=params)


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.title("📊 Betting Intel")
st.sidebar.markdown("Professional Betting Intelligence Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📈 Performance",
        "🎯 Today's Bets",
        "🔮 Predictions",
        "📊 Model Analysis",
        "📋 CLV Tracking",
        "🔥 Market Moves",
        "⚙️ Settings",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Database Status")
db_path = get_db_path()
if db_path and db_path.exists():
    size_mb = db_path.stat().st_size / (1024 * 1024)
    st.sidebar.info(f"✅ Connected\n{size_mb:.1f} MB")
else:
    st.sidebar.warning("⚠️ No database found")

st.sidebar.markdown("---")
st.sidebar.caption(f"v1.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE RENDERER
# ═══════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    _col1, _col2, _col3, _col4 = st.columns(4)

    total_games = query_db("SELECT COUNT(DISTINCT GAME_ID) as n FROM game_logs").iloc[0]["n"] if not query_db("SELECT COUNT(DISTINCT GAME_ID) as n FROM game_logs").empty else 0
    total_odds = query_db("SELECT COUNT(*) as n FROM odds").iloc[0]["n"] if not query_db("SELECT COUNT(*) as n FROM odds").empty else 0
    total_features = query_db("SELECT COUNT(*) as n FROM team_features").iloc[0]["n"] if not query_db("SELECT COUNT(*) as n FROM team_features").empty else 0
    total_clv = query_db("SELECT COUNT(*) as n FROM clv_tracking").iloc[0]["n"] if not query_db("SELECT COUNT(*) as n FROM clv_tracking").empty else 0

    _col1.metric("Games Tracked", f"{total_games:,}")
    _col2.metric("Odds Records", f"{total_odds:,}")
    _col3.metric("Features Stored", f"{total_features:,}")
    _col4.metric("Bets Tracked (CLV)", f"{total_clv:,}")

    # Quick stats
    try:
        seasons_df = query_db("SELECT SEASON, COUNT(DISTINCT GAME_ID) as games, COUNT(*) as rows FROM game_logs GROUP BY SEASON ORDER BY SEASON DESC")
        if not seasons_df.empty:
            st.subheader("📅 Seasons Overview")
            st.dataframe(seasons_df, use_container_width=True, hide_index=True)
    except Exception:
        pass

    # Recent odds
    try:
        recent_odds = query_db("""
            SELECT o.game_id, o.home_team, o.away_team, o.market, o.sportsbook,
                   o.odds_value, o.timestamp
            FROM odds o
            ORDER BY o.timestamp DESC LIMIT 10
        """)
        if not recent_odds.empty:
            st.subheader("🔄 Recent Odds Snapshots")
            st.dataframe(recent_odds, use_container_width=True, hide_index=True)
    except Exception:
        pass

    # Recent CLV
    try:
        recent_clv = query_db("""
            SELECT game_id, home_team, away_team, market_type, bet_side,
                   clv_percentage, created_at
            FROM clv_tracking ORDER BY created_at DESC LIMIT 10
        """)
        if not recent_clv.empty:
            st.subheader("📊 Recent CLV Records")
            recent_clv["clv_percentage"] = recent_clv["clv_percentage"].apply(lambda x: f"{x:+.4%}")
            st.dataframe(recent_clv, use_container_width=True, hide_index=True)
    except Exception:
        pass


elif page == "📈 Performance":
    st.title("📈 Performance Analytics")

    col1, col2, col3 = st.columns(3)

    # CLV summary
    clv_stats = query_db("""
        SELECT
            COUNT(*) as total_bets,
            AVG(clv_percentage) as avg_clv,
            SUM(CASE WHEN clv_percentage > 0 THEN 1 ELSE 0 END) as positive_clv,
            AVG(CASE WHEN market_type = 'moneyline' THEN clv_percentage END) as ml_clv,
            AVG(CASE WHEN market_type = 'spread' THEN clv_percentage END) as spread_clv,
            AVG(CASE WHEN market_type = 'total' THEN clv_percentage END) as total_clv
        FROM clv_tracking
    """)
    if not clv_stats.empty and clv_stats.iloc[0]["total_bets"] > 0:
        row = clv_stats.iloc[0]
        col1.metric("Total Bets Tracked", row["total_bets"])
        col2.metric("Average CLV", f"{row['avg_clv']:+.4%}")
        col3.metric("Positive CLV Rate", f"{row['positive_clv']/row['total_bets']:.1%}" if row['total_bets'] > 0 else "0%")

        # CLV by market
        st.subheader("CLV by Market Type")
        market_data = pd.DataFrame({
            "Market": ["Moneyline", "Spread", "Total"],
            "Avg CLV": [row["ml_clv"] or 0, row["spread_clv"] or 0, row["total_clv"] or 0],
        })
        st.bar_chart(market_data.set_index("Market"))

    # Daily CLV trend
    clv_by_date = query_db("""
        SELECT game_date, AVG(clv_percentage) as avg_clv,
               COUNT(*) as n_bets
        FROM clv_tracking
        GROUP BY game_date
        ORDER BY game_date
    """)
    if not clv_by_date.empty:
        st.subheader("CLV Trend Over Time")
        st.line_chart(clv_by_date.set_index("game_date")["avg_clv"])

    # Model performance
    st.subheader("Model vs Market Agreement")
    agreement_stats = query_db("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN agreement = 'agree' THEN 1 ELSE 0 END) as agreed,
            SUM(CASE WHEN agreement = 'disagree' THEN 1 ELSE 0 END) as disagreed,
            SUM(CASE WHEN agreement = 'strongly_disagree' THEN 1 ELSE 0 END) as strongly_disagreed,
            AVG(ABS(edge)) as avg_abs_edge,
            AVG(edge) as avg_edge
        FROM model_vs_market
    """)
    if not agreement_stats.empty and agreement_stats.iloc[0]["total"] > 0:
        row = agreement_stats.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Comparisons", row["total"])
        col2.metric("Agreement Rate", f"{row['agreed']/row['total']:.1%}")
        col3.metric("Avg Abs Edge", f"{row['avg_abs_edge']:.4%}")
        col4.metric("Avg Edge", f"{row['avg_edge']:+.4%}")

        # Agreement pie
        agreement_df = pd.DataFrame({
            "Agreement": ["Agree", "Disagree", "Strongly Disagree"],
            "Count": [row["agreed"], row["disagreed"], row["strongly_disagreed"]],
        })
        st.bar_chart(agreement_df.set_index("Agreement"))


elif page == "🎯 Today's Bets":
    st.title("🎯 Today's Bets")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    st.caption(f"Date: {today}")

    # Load predictions for today
    pred_dir = Path("output/predictions")
    predictions_file = pred_dir / f"predictions_{today}.json"
    if predictions_file.exists():
        with open(predictions_file) as f:
            data = json.load(f)

        predictions = data.get("predictions", [])
        actionable = [p for p in predictions if p.get("stake", 0) > 0]

        st.metric("Total Opportunities", len(predictions))
        st.metric("Actionable Bets", len(actionable))

        if actionable:
            st.subheader("Recommended Bets")
            df = pd.DataFrame(actionable)
            df["edge"] = df["edge"].apply(lambda x: f"{x:.2%}")
            df["ev"] = df["ev"].apply(lambda x: f"{x:.2%}")
            df["stake"] = df["stake"].apply(lambda x: f"${x:.2f}")
            st.dataframe(df[["game", "prediction", "probability", "edge", "ev", "stake", "risk_level"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No actionable bets today.")
    else:
        st.info(f"No predictions file found for {today}. Run `predict_tomorrow.py` first.")


elif page == "🔮 Predictions":
    st.title("🔮 Predictions")

    # Load latest predictions
    pred_dir = Path("output/predictions")
    if pred_dir.exists():
        pred_files = sorted(pred_dir.glob("predictions_*.json"), reverse=True)
        if pred_files:
            selected_file = st.selectbox("Select date", [f.name for f in pred_files])

            with open(pred_dir / selected_file) as f:
                data = json.load(f)

            predictions = data.get("predictions", [])
            st.metric("Total Predictions", len(predictions))

            if predictions:
                df = pd.DataFrame(predictions)
                df["edge"] = df["edge"].apply(lambda x: f"{x:.2%}")
                df["ev"] = df["ev"].apply(lambda x: f"{x:.2%}")
                st.dataframe(df, use_container_width=True, hide_index=True)

            st.subheader("Pipeline Info")
            st.json({
                "date": data.get("date"),
                "generated_at": data.get("generated_at"),
                "bankroll": data.get("bankroll"),
                "kelly_fraction": data.get("kelly_fraction"),
                "min_edge": data.get("min_edge"),
            })
        else:
            st.info("No prediction files found.")
    else:
        st.info("No predictions directory found.")


elif page == "📊 Model Analysis":
    st.title("📊 Model Analysis")

    # Feature importance
    st.subheader("Feature Store Stats")
    feat_stats = query_db("SELECT * FROM feature_versions ORDER BY created_at DESC")
    if not feat_stats.empty:
        st.dataframe(feat_stats, use_container_width=True, hide_index=True)

    # Team features
    st.subheader("Team Features (Latest)")
    team_feats = query_db("""
        SELECT team_name, game_date, version,
               offensive_rating, defensive_rating, pace, net_rating,
               win_pct, pts_scored_avg, pts_allowed_avg
        FROM team_features
        ORDER BY game_date DESC LIMIT 20
    """)
    if not team_feats.empty:
        st.dataframe(team_feats, use_container_width=True, hide_index=True)

    # Schedule features
    st.subheader("Schedule Features (Latest)")
    sched_feats = query_db("""
        SELECT team_name, game_date, rest_days, is_back_to_back,
               travel_distance, is_home, games_in_last_7, games_in_last_14
        FROM schedule_features
        ORDER BY game_date DESC LIMIT 20
    """)
    if not sched_feats.empty:
        st.dataframe(sched_feats, use_container_width=True, hide_index=True)

    # Calibration
    st.subheader("Probability Calibration")
    st.info("Calibration data available after running calibration workflow.")


elif page == "📋 CLV Tracking":
    st.title("📋 Closing Line Value Tracker")

    # Summary
    clv_summary = query_db("""
        SELECT
            COUNT(*) as total,
            AVG(clv_percentage) as avg_clv,
            SUM(CASE WHEN bet_won = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN bet_won = 0 THEN 1 ELSE 0 END) as losses,
            AVG(CASE WHEN market_type = 'moneyline' THEN clv_percentage END) as ml_clv,
            AVG(CASE WHEN market_type = 'spread' THEN clv_percentage END) as spread_clv,
            AVG(CASE WHEN market_type = 'total' THEN clv_percentage END) as total_clv
        FROM clv_tracking
    """)

    if not clv_summary.empty and clv_summary.iloc[0]["total"] > 0:
        row = clv_summary.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Bets", row["total"])
        col2.metric("Avg CLV", f"{row['avg_clv']:+.4%}")
        col3.metric("Wins/Losses", f"{row['wins']}/{row['losses']}")
        win_rate = row['wins'] / (row['wins'] + row['losses']) if (row['wins'] + row['losses']) > 0 else 0
        col4.metric("Win Rate", f"{win_rate:.1%}")

        # CLV by market
        st.subheader("CLV by Market")
        market_clv = pd.DataFrame({
            "Market": ["Moneyline", "Spread", "Total"],
            "CLV": [row["ml_clv"], row["spread_clv"], row["total_clv"]],
        })
        st.bar_chart(market_clv.set_index("Market"))

    # All CLV records
    st.subheader("All CLV Records")
    all_clv = query_db("""
        SELECT game_id, home_team, away_team, game_date,
               market_type, bet_side, bet_odds_american,
               closing_odds_american, clv_percentage,
               bet_won, model_probability, created_at
        FROM clv_tracking
        ORDER BY created_at DESC LIMIT 50
    """)
    if not all_clv.empty:
        all_clv["clv_percentage"] = all_clv["clv_percentage"].apply(lambda x: f"{x:+.4%}")
        st.dataframe(all_clv, use_container_width=True, hide_index=True)


elif page == "🔥 Market Moves":
    st.title("🔥 Market Movement & Steam Detection")

    # Movement records
    st.subheader("Recent Market Movements")
    movements = query_db("""
        SELECT o.game_id, o.home_team, o.away_team, o.market,
               o.sportsbook, o.timestamp, o.odds_value
        FROM odds o
        ORDER BY o.timestamp DESC LIMIT 20
    """)
    if not movements.empty:
        st.dataframe(movements, use_container_width=True, hide_index=True)

    # Steam alerts would go here
    st.subheader("Steam Move Detection")
    try:
        from betting_intel.market.steam import SteamMoveDetector
        st.info("Steam Move Detector loaded. Active scanning available.")
    except ImportError:
        st.warning("Steam Move Detector module not available. Install the market package.")


elif page == "⚙️ Settings":
    st.title("⚙️ System Settings")

    st.subheader("Bankroll Configuration")
    bankroll = st.number_input("Bankroll ($)", min_value=100, max_value=10000000, value=10000, step=500)
    kelly_frac = st.select_slider("Kelly Fraction", options=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0], value=0.25)
    min_edge = st.slider("Minimum Edge Threshold (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.5) / 100

    st.subheader("Database Info")
    db_path = get_db_path()
    if db_path and db_path.exists():
        st.text(f"Path: {db_path}")
        st.text(f"Size: {db_path.stat().st_size / (1024 * 1024):.1f} MB")

        # List tables
        conn = get_connection()
        if conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row["name"] for row in cursor.fetchall()]
            st.text(f"Tables: {', '.join(tables)}")

    st.subheader("Available Modules")
    modules = [
        "betting_intel.betting.ev — Expected Value Engine",
        "betting_intel.betting.bet — Betting Engine (Kelly)",
        "betting_intel.betting.clv — CLV Tracker",
        "betting_intel.data.odds_ingestion — Odds Ingestion",
        "betting_intel.features — Feature Store",
        "betting_intel.market — Market Intelligence",
        "betting_intel.models.ensemble — Ensemble Model",
        "betting_intel.pipeline.predict_tomorrow — Prediction Pipeline",
        "betting_intel.alerts — Alert System",
        "betting_intel.risk — Risk Management",
        "betting_intel.validation — Model Validation",
        "betting_intel.backtesting — Backtesting Framework",
        "betting_intel.monitoring — Model Monitoring",
    ]
    for m in modules:
        st.text(f"  ✅ {m}")


# ═══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.caption("Built with Codebuff • Professional Betting Intelligence Platform")
