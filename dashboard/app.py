"""
 EXACT BET RECOMMENDATIONS DASHBOARD

Shows only what matters: exactly what to bet, how much, and why.
No probabilities, no confusion — just clear, actionable bets.

Run: streamlit run dashboard/app.py
"""

import sys
import os
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR

st.set_page_config(
    page_title="Betting Intelligence - EXACT BETS",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00ff88 0%, #00ccff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subheader { font-size: 1rem; color: #888; margin-bottom: 1rem; }

    .clear-pick-badge {
        background: linear-gradient(135deg, #00ff88 0%, #00cc88 100%);
        color: #000;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.7rem;
        display: inline-block;
    }

    .bet-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4e;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s;
    }
    .bet-card:hover {
        border-color: #00ff88;
        box-shadow: 0 0 20px rgba(0,255,136,0.1);
        transform: translateY(-2px);
    }
    .bet-card.highlight {
        border-color: #00ff88;
        background: linear-gradient(135deg, #1a2e1a 0%, #1a1a2e 100%);
    }
    .bet-card .action-text {
        font-size: 1.3rem;
        font-weight: 800;
        color: #fff;
    }
    .bet-card .action-stake {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00ff88;
    }
    .bet-card .bet-meta { font-size: 0.75rem; color: #888; }
    .bet-card .edge-pos { color: #00ff88; font-weight: 700; }
    .bet-card .reasoning { font-size: 0.8rem; color: #aaa; margin-top: 4px; }

    .metric-tile {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4e;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-tile .value { font-size: 1.8rem; font-weight: 800; }
    .metric-tile .label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-tile .delta-pos { color: #00ff88; }
    .metric-tile .delta-neg { color: #ff4444; }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ddd;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #2a2a4e;
        padding-bottom: 0.5rem;
    }

    .bet-amount {
        font-size: 1.5rem;
        font-weight: 800;
        color: #00ccff;
    }

    .badge-nba { background: #17408B; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; }
    .badge-lnb { background: #002654; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; }
    .badge-cebl { background: #CE1126; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; }
    .badge-bnxt { background: #FF6600; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; }

    .conf-VERY_HIGH { color: #00ff88; font-weight: 700; }
    .conf-HIGH { color: #00ccff; font-weight: 600; }
    .conf-MEDIUM { color: #ffaa00; }
    .conf-LOW { color: #ff6644; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_recommendations() -> tuple:
    """Load all recommendations via the recommendation engine."""
    from betting_intel.recommendations import RecommendationEngine
    engine = RecommendationEngine()
    all_bets = engine.generate_all_bets()
    clear_picks = engine.get_clear_picks()
    summary = engine.get_summary()
    todays = engine.get_todays_card()
    return all_bets, clear_picks, summary, todays


# ── Helper Functions ──────────────────────────────────────────────────────

def badge_html(league: str) -> str:
    badge_map = {"nba": "badge-nba", "lnb_pro_b": "badge-lnb", "cebl": "badge-cebl", "bnxt": "badge-bnxt"}
    cls = badge_map.get(league.lower(), "badge-nba")
    return f'<span class="{cls}">{league.upper()}</span>'


def render_bet_action(bet, highlight: bool = False) -> str:
    """Render a bet as a single actionable line — WHAT and HOW MUCH."""
    hl = "highlight" if highlight else ""
    badge = badge_html(bet.league)
    cp_badge = "★ CLEAR PICK " if bet.is_clear_pick else ""

    action_text = bet.action.replace(" on ", "\n    on\n")
    edge_str = f"<span class='edge-pos'>+{bet.edge_pct:.1%} edge</span>" if bet.edge_pct > 0 else ""

    return f"""
    <div class="bet-card {hl}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex: 1;">
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    {f'<span class="clear-pick-badge">{cp_badge}</span>' if bet.is_clear_pick else ''}
                    <span style="font-size:0.75rem; color:#888; margin-left: 8px;">
                        {badge} · {bet.matchup} · {bet.game_date}
                    </span>
                </div>
                <div class="action-text">
                    ${bet.stake_dollars:.0f} <span style="color: #aaa; font-weight: 400;">on</span> {bet.bet_side}
                </div>
                <div class="reasoning">{bet.reasoning[:150]}</div>
            </div>
            <div style="text-align: right; min-width: 120px;">
                <div class="action-stake">${bet.stake_dollars:.0f}</div>
                <div style="font-size:0.7rem; color:#888;">Stake</div>
                <div style="margin-top:4px;">{edge_str}</div>
            </div>
        </div>
    </div>
    """


def render_clear_pick_card(cp, idx: int) -> str:
    """Render a clear pick as a prominent, actionable bet card."""
    bet = cp.bet
    badge = badge_html(bet.league)
    risk_color = {"CONSERVATIVE": "#00ff88", "MODERATE": "#ffaa00", "AGGRESSIVE": "#ff4444"}.get(cp.risk_level, "#888")
    reasons_html = "".join([f'<div style="font-size:0.75rem;color:#aaa;margin-top:2px;">→ {r}</div>' for r in cp.reasons])

    return f"""
    <div class="bet-card highlight">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div>
                <span class="clear-pick-badge">★ PLACE THIS BET #{idx}</span>
                <span style="font-size:0.7rem;color:#888;margin-left:8px;">
                    {cp.clear_score:.0f}/100 confidence
                </span>
                <span style="font-size:0.7rem;color:{risk_color};margin-left:8px;font-weight:600;">
                    {cp.risk_level}
                </span>
                <span style="margin-left:8px;">{badge}</span>
            </div>
            <div style="font-size:0.75rem;color:#888;">{bet.matchup} · {bet.game_date}</div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="flex: 2;">
                <div class="action-text" style="font-size:1.5rem;">
                    ${bet.stake_dollars:.0f} <span style="color:#aaa;font-weight:400;">on</span> {bet.bet_side}
                </div>
                <div style="font-size:0.75rem;color:#666;margin-top:4px;">
                    {bet.bet_type.display_name()} · {bet.league}
                </div>
                <div style="margin-top: 4px;">{reasons_html}</div>
            </div>
            <div style="text-align: center; min-width: 100px;">
                <div style="font-size:1.5rem;font-weight:700;color:#00ff88;">
                    +{bet.edge_pct:.1%}
                </div>
                <div style="font-size:0.7rem;color:#888;">Edge</div>
            </div>
            <div style="text-align: center; min-width: 100px;">
                <div style="font-size:1.5rem;font-weight:700;color:#00ccff;">
                    ${bet.stake_dollars:.0f}
                </div>
                <div style="font-size:0.7rem;color:#888;">Stake</div>
            </div>
        </div>
    </div>
    """


# ── Main App ──────────────────────────────────────────────────────────────

def main():
    st.markdown('<div class="main-header"> EXACT BETS</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subheader">Place These Bets · {datetime.now().strftime("%A, %B %d, %Y · %I:%M %p")}</div>', unsafe_allow_html=True)

    # ── Load Data ──────────────────────────────────────────────────────
    with st.spinner("Analyzing markets and generating exact bets..."):
        try:
            all_bets, clear_picks, summary, todays_bets = load_recommendations()
        except Exception as e:
            st.error(f"Could not load recommendations: {e}")
            st.info("Make sure the package is installed: `pip install -e .` from the betting-intelligence directory")
            st.stop()

    # ── Sidebar ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("###  Filters")

        leagues = ["all"] + sorted(set(b.league for b in all_bets))
        selected_league = st.selectbox("League", leagues, index=0)

        bet_types = ["all"] + sorted(set(b.bet_type.value for b in all_bets))
        bt_display = {"all": "All Types", "moneyline": "Moneyline", "spread": "Spread",
                       "total_points": "Total O/U", "team_total": "Team Total",
                       "first_quarter_winner": "1st Quarter", "first_half_total": "1st Half Total",
                       "player_points": "Player Points", "player_rebounds": "Player Rebounds",
                       "player_assists": "Player Assists", "player_pra": "Player PRA"}
        bt_labels = [bt_display.get(bt, bt) for bt in bet_types]
        selected_bt_label = st.selectbox("Bet Type", bt_labels, index=0)
        selected_bt = [k for k, v in bt_display.items() if v == selected_bt_label][0] if selected_bt_label != "All Types" else "all"

        min_edge = st.slider("Min Edge %", 0.0, 10.0, 1.0, 0.5) / 100
        clear_only = st.toggle("Clear Picks Only", value=False)

        st.markdown("---")
        st.markdown("###  Summary")

        col1, col2 = st.columns(2)
        with col1: st.metric("Clear Picks", summary['clear_picks'])
        with col2: st.metric("Avg Edge", f"{summary['avg_edge']:.1%}")

        games_count = len(set(b.matchup for b in todays_bets))
        st.metric("Games Available", games_count)
        st.metric("Total Stake", f"${summary['total_stake']:,.0f}")

        if st.button(" Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Filter Bets ─────────────────────────────────────────────────────
    filtered_bets = all_bets.copy()
    if selected_league != "all":
        filtered_bets = [b for b in filtered_bets if b.league.lower() == selected_league.lower()]
    if selected_bt != "all":
        filtered_bets = [b for b in filtered_bets if b.bet_type.value == selected_bt]
    filtered_bets = [b for b in filtered_bets if b.edge_pct >= min_edge]
    if clear_only:
        filtered_bets = [b for b in filtered_bets if b.is_clear_pick]
    filtered_bets.sort(key=lambda b: b.edge_pct, reverse=True)

    filtered_clear = [
        cp for cp in clear_picks
        if (selected_league == "all" or cp.bet.league.lower() == selected_league.lower())
        and (selected_bt == "all" or cp.bet.bet_type.value == selected_bt)
        and cp.bet.edge_pct >= min_edge
    ]

    # ── Main Content ──────────────────────────────────────────────────
    tab_clear, tab_today, tab_all = st.tabs(["★ PLACE THESE BETS", " Today's Card", " All Bets"])

    # ── TAB 1: CLEAR PICKS — THE EXACT BETS TO PLACE ───────────────────
    with tab_clear:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0a2e1a 0%, #1a1a2e 100%);
                    border: 2px solid #00ff88; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;">
            <div style="font-size:1.1rem; font-weight:700; color:#00ff88;">
                 HIGH-CONFIDENCE BETS — PLACE THESE
            </div>
            <div style="font-size:0.85rem; color:#aaa; margin-top:4px;">
                {len(filtered_clear)} picks that meet strict criteria: ≥3% edge, ≥55% win prob, HIGH+ confidence
            </div>
        </div>
        """, unsafe_allow_html=True)

        if filtered_clear:
            for i, cp in enumerate(filtered_clear, 1):
                st.markdown(render_clear_pick_card(cp, i), unsafe_allow_html=True)
                st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

            total_stake = sum(c.bet.stake_dollars for c in filtered_clear)
            avg_edge = sum(c.bet.edge_pct for c in filtered_clear) / len(filtered_clear)

            st.markdown(f"""
            <div style="background:#1a1a2e;border:1px solid #2a2a4e;border-radius:12px;padding:1.5rem;margin-top:1rem;
                        text-align:center;">
                <div style="font-size:0.8rem;color:#888;text-transform:uppercase;letter-spacing:1px;">Total to Bet</div>
                <div style="font-size:2.5rem;font-weight:800;color:#00ff88;margin:4px 0;">${total_stake:.0f}</div>
                <div style="font-size:0.85rem;color:#aaa;">
                    {len(filtered_clear)} bets · Avg edge: {avg_edge:.1%} · Risk: {
                        "Balanced" if any(c.risk_level == "CONSERVATIVE" for c in filtered_clear) else "Aggressive"
                    }
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No clear picks match your filters. Try lowering the edge threshold.")

    # ── TAB 2: TODAY'S CARD ────────────────────────────────────────────
    with tab_today:
        today = date.today()
        valid_dates = [today.isoformat()]
        for d in [today + timedelta(days=1), today + timedelta(days=2)]:
            valid_dates.append(d.isoformat())
        today_bets = [b for b in filtered_bets if b.game_date in valid_dates]
        if not today_bets:
            today_bets = filtered_bets[:20]

        games = {}
        for bet in today_bets:
            key = bet.matchup
            if key not in games:
                games[key] = {"league": bet.league, "bets": []}
            games[key]["bets"].append(bet)

        if games:
            for matchup, game_info in games.items():
                game_bets = game_info["bets"]
                league = game_info["league"]
                clear_count = sum(1 for b in game_bets if b.is_clear_pick)
                badge = badge_html(league)
                cp_tag = f'<span class="clear-pick-badge" style="margin-left:8px;">{clear_count} picks</span>' if clear_count else ""

                st.markdown(f'<div class="section-title">{matchup} {badge} {cp_tag}</div>', unsafe_allow_html=True)
                for bet in game_bets:
                    st.markdown(render_bet_action(bet, highlight=bet.is_clear_pick), unsafe_allow_html=True)
        else:
            st.info("No games found.")

    # ── TAB 3: ALL BETS ─────────────────────────────────────────────────
    with tab_all:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-tile"><div class="value" style="color:#00ff88;">{len(filtered_bets)}</div><div class="label">Bets Shown</div></div>', unsafe_allow_html=True)
        with col2:
            edges = [b.edge_pct for b in filtered_bets]
            avg_e = np.mean(edges) if edges else 0
            st.markdown(f'<div class="metric-tile"><div class="value" style="color:#00ccff;">{avg_e:.1%}</div><div class="label">Avg Edge</div></div>', unsafe_allow_html=True)
        with col3:
            total_s = sum(b.stake_dollars for b in filtered_bets)
            st.markdown(f'<div class="metric-tile"><div class="value" style="color:#FF69B4;">${total_s:,.0f}</div><div class="label">Total Stake</div></div>', unsafe_allow_html=True)
        with col4:
            high_conf = sum(1 for b in filtered_bets if b.is_clear_pick)
            st.markdown(f'<div class="metric-tile"><div class="value" style="color:#9B59B6;">{high_conf}</div><div class="label">Clear Picks</div></div>', unsafe_allow_html=True)

        if filtered_bets:
            for bet in filtered_bets:
                st.markdown(render_bet_action(bet, highlight=bet.is_clear_pick), unsafe_allow_html=True)
        else:
            st.info("No bets match your filters.")

    # ── Footer ─────────────────────────────────────────────────────────
    st.markdown("---")
    # Summary bar: total stake info
    total_all_stake = sum(b.stake_dollars for b in all_bets)
    st.caption(f"Betting Intelligence System v0.2.0 · Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
               f"{len(clear_picks)} clear picks · "
               f"Total exposure: ${summary['total_stake']:,.0f} on $10,000 bankroll")


if __name__ == "__main__":
    main()
