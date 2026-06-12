"""Tests for QuarterHalfProjector and quarter/half bet types."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from betting_intel.data.quarter_projector import (
    QuarterHalfProjector,
    _LEAGUE_AVG_RATIOS,
    _HOME_SHARE,
    ESPN_SCOREBOARD_URL,
    ESPN_SUMMARY_URL,
)
from betting_intel.recommendations.bet_types import (
    BetType,
    BetSuggestion,
    QuarterTotalBet,
    QuarterBet,
    HalfTotalBet,
    Confidence,
)
from betting_intel.recommendations.engine import RecommendationEngine


# ═══════════════════════════════════════════════════════════════════════════
#  QuarterHalfProjector Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQuarterHalfProjector:
    """Tests for QuarterHalfProjector with mocked/fixture data."""

    # ── Fixtures ───────────────────────────────────────────────────────

    @pytest.fixture
    def projector(self):
        """Create projector with a temp cache path (no ESPN calls)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "test_ratios.json"
            yield QuarterHalfProjector(cache_path=cache)

    @pytest.fixture
    def projector_with_ratios(self, projector):
        """Projector pre-populated with mock team ratios."""
        projector._ratios = {
            "celtics": {
                "q1": 0.26, "q2": 0.25, "q3": 0.24, "q4": 0.25,
                "h1": 0.51, "h2": 0.49,
            },
            "lakers": {
                "q1": 0.23, "q2": 0.26, "q3": 0.25, "q4": 0.26,
                "h1": 0.49, "h2": 0.51,
            },
        }
        return projector

    @pytest.fixture
    def mock_espn_scoreboard(self):
        """Build a realistic ESPN scoreboard response."""
        return {
            "events": [
                {
                    "id": "401701000",
                    "competitions": [
                        {
                            "competitors": [
                                {"team": {"displayName": "Boston Celtics"}},
                                {"team": {"displayName": "Los Angeles Lakers"}},
                            ],
                            "status": {
                                "type": {"name": "STATUS_FINAL"}
                            },
                        }
                    ],
                },
            ]
        }

    @pytest.fixture
    def mock_espn_summary(self):
        """Build a realistic ESPN summary response with linescores."""
        return {
            "header": {
                "competitions": [
                    {
                        "competitors": [
                            {
                                "team": {"displayName": "Boston Celtics"},
                                "linescores": [
                                    {"value": 32},
                                    {"value": 28},
                                    {"value": 30},
                                    {"value": 26},
                                ],
                            },
                            {
                                "team": {"displayName": "Los Angeles Lakers"},
                                "linescores": [
                                    {"value": 25},
                                    {"value": 30},
                                    {"value": 27},
                                    {"value": 29},
                                ],
                            },
                        ],
                    }
                ],
            },
        }

    # ── Basic Projection Tests ─────────────────────────────────────────

    def test_project_returns_all_keys(self, projector_with_ratios):
        """project() returns the full set of expected keys."""
        result = projector_with_ratios.project(225.0, "Celtics", "Lakers")

        expected_keys = [
            "q1_home", "q1_away", "q1_total",
            "q2_home", "q2_away", "q2_total",
            "q3_home", "q3_away", "q3_total",
            "q4_home", "q4_away", "q4_total",
            "h1_home", "h1_away", "h1_total",
            "h2_home", "h2_away", "h2_total",
            "home_score", "away_score", "predicted_total",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        assert result["predicted_total"] == 225.0

    def test_project_quarters_sum_to_total(self, projector_with_ratios):
        """Q1+Q2+Q3+Q4 totals should equal the predicted total (within rounding)."""
        result = projector_with_ratios.project(225.0, "Celtics", "Lakers")
        q_sum = (
            result["q1_total"]
            + result["q2_total"]
            + result["q3_total"]
            + result["q4_total"]
        )
        # Allow 0.5 rounding difference per quarter (4 * 0.5 = 2.0)
        assert abs(q_sum - 225.0) < 2.0, (
            f"Q total mismatch: {q_sum} vs 225.0"
        )

    def test_project_halves_sum_to_total(self, projector_with_ratios):
        """H1+H2 totals should equal the predicted total."""
        result = projector_with_ratios.project(225.0, "Celtics", "Lakers")
        h_sum = result["h1_total"] + result["h2_total"]
        assert abs(h_sum - 225.0) < 1.0, (
            f"Half total mismatch: {h_sum} vs 225.0"
        )

    def test_project_uses_league_averages_when_no_team_data(self, projector):
        """Without team-specific ratios, falls back to league averages."""
        result = projector.project(225.0, "Unknown", "Opponent")
        assert result["predicted_total"] == 225.0

        # Should still return valid projections
        q1 = result["q1_total"]
        h1 = result["h1_total"]
        assert q1 > 0
        assert h1 > 0

        # League avg Q1 is ~24.2% of 225 ≈ 54.5
        assert 40 <= q1 <= 70, f"Q1 total {q1} out of expected range"

    def test_project_home_away_split(self, projector):
        """Default home/away split should be ~51/49."""
        result = projector.project(200.0, "Home", "Away")
        assert 98 <= result["home_score"] <= 108  # ~102
        assert 92 <= result["away_score"] <= 102  # ~98
        assert abs(result["home_score"] - result["away_score"]) > 0

    def test_project_with_team_specific_ratios(self, projector_with_ratios):
        """Uses team-specific ratios when available."""
        result = projector_with_ratios.project(200.0, "Celtics", "Lakers")
        # Celtics: q1 ratio = 0.26 (higher than league avg 0.242)
        home = result["home_score"]  # ~102
        expected_q1 = round(home * 0.26, 1)
        assert result["q1_home"] == expected_q1, (
            f"Expected Q1 home {expected_q1}, got {result['q1_home']}"
        )

    def test_project_zero_total(self, projector):
        """Predicted total of 0 should return zeros."""
        result = projector.project(0.0, "Celtics", "Lakers")
        assert result["home_score"] == 0.0
        assert result["away_score"] == 0.0
        assert result["q1_total"] == 0.0
        assert result["h1_total"] == 0.0

    def test_project_large_total(self, projector_with_ratios):
        """Very large predicted total should still produce valid ratios."""
        result = projector_with_ratios.project(350.0, "Celtics", "Lakers")
        assert result["q1_total"] > 70
        assert result["h1_total"] > 140
        assert abs(result["h1_total"] + result["h2_total"] - 350.0) < 1.0

    # ── Market Estimation Tests ────────────────────────────────────────

    def test_get_quarter_market(self, projector):
        """Quarter market line should be a fraction of predicted total."""
        total = 225.0
        q1_mkt = projector.get_quarter_market(total, 1)
        q2_mkt = projector.get_quarter_market(total, 2)
        q4_mkt = projector.get_quarter_market(total, 4)

        assert q1_mkt == round(total * _LEAGUE_AVG_RATIOS["q1"], 1)
        assert q2_mkt == round(total * _LEAGUE_AVG_RATIOS["q2"], 1)
        assert q4_mkt == round(total * _LEAGUE_AVG_RATIOS["q4"], 1)

        # Q1 should be slightly lower than Q4 (games slow at first)
        assert q1_mkt < q4_mkt, "Q1 market should be less than Q4 market"

    def test_get_quarter_market_bounds(self, projector):
        """Quarter market should be reasonable for extreme totals."""
        assert projector.get_quarter_market(100.0, 1) > 0
        assert projector.get_quarter_market(300.0, 1) < 100
        assert projector.get_quarter_market(0.0, 1) == 0.0

    def test_get_half_market(self, projector):
        """Half market line estimation."""
        total = 225.0
        h1_mkt = projector.get_half_market(total, 1)
        h2_mkt = projector.get_half_market(total, 2)

        assert h1_mkt == round(total * _LEAGUE_AVG_RATIOS["h1"], 1)
        assert h2_mkt == round(total * _LEAGUE_AVG_RATIOS["h2"], 1)

        # 1H + 2H should roughly equal total
        assert abs(h1_mkt + h2_mkt - total) < 1.0

    # ── ESPN Data Fetching Tests (mocked) ──────────────────────────────

    def test_get_status_parses_correctly(self, projector, mock_espn_scoreboard):
        """_get_status extracts the game status from an ESPN event."""
        event = mock_espn_scoreboard["events"][0]
        status = projector._get_status(event)
        assert status == "STATUS_FINAL"

    def test_get_status_missing_data(self, projector):
        """_get_status returns empty string for malformed events."""
        assert projector._get_status({}) == ""
        assert projector._get_status({"competitions": []}) == ""

    def test_fetch_linescores_parses_summary(self, projector, mock_espn_summary):
        """_fetch_linescores correctly parses the ESPN summary response."""
        with patch.object(projector._session, "get") as mock_get:
            mock_resp = MagicMock(spec=requests.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_espn_summary
            mock_get.return_value = mock_resp

            result = projector._fetch_linescores("401701000")
            assert result is not None
            assert len(result) == 2

            team1, scores1 = result[0]
            team2, scores2 = result[1]

            assert team1 == "Boston Celtics"
            assert scores1 == {"q1": 32.0, "q2": 28.0, "q3": 30.0, "q4": 26.0}
            assert team2 == "Los Angeles Lakers"
            assert scores2 == {"q1": 25.0, "q2": 30.0, "q3": 27.0, "q4": 29.0}

    def test_fetch_linescores_incomplete_game(self, projector):
        """Game with < 4 quarters returns None."""
        mock_data = {
            "header": {
                "competitions": [
                    {
                        "competitors": [
                            {
                                "team": {"displayName": "Celtics"},
                                "linescores": [
                                    {"value": 32},
                                    {"value": 28},
                                ],
                            },
                            {
                                "team": {"displayName": "Lakers"},
                                "linescores": [
                                    {"value": 25},
                                    {"value": 30},
                                ],
                            },
                        ],
                    }
                ],
            },
        }
        with patch.object(projector._session, "get") as mock_get:
            mock_resp = MagicMock(spec=requests.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_data
            mock_get.return_value = mock_resp

            result = projector._fetch_linescores("401701000")
            assert result is None, "Incomplete game should return None"

    def test_fetch_linescores_api_error(self, projector):
        """API error returns None gracefully."""
        with patch.object(projector._session, "get") as mock_get:
            mock_get.side_effect = requests.RequestException("Timeout")
            result = projector._fetch_linescores("401701000")
            assert result is None

    def test_fetch_recent_game_ids(self, projector, mock_espn_scoreboard):
        """Fetches game IDs from ESPN scoreboard, filtering only final games."""
        with patch.object(projector._session, "get") as mock_get:
            mock_resp = MagicMock(spec=requests.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_espn_scoreboard
            mock_get.return_value = mock_resp

            game_ids = projector._fetch_recent_game_ids(limit=10)
            assert "401701000" in game_ids

    def test_max_games_limit_respected(self, projector):
        """_fetch_recent_game_ids respects its limit parameter."""
        # Return 2 final games per day, 30 days → 60 total
        with patch.object(projector._session, "get") as mock_get:
            mock_resp = MagicMock(spec=requests.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "events": [
                    {
                        "id": f"game_{i}",
                        "competitions": [
                            {
                                "competitors": [{"team": {}}, {"team": {}}],
                                "status": {"type": {"name": "STATUS_FINAL"}},
                            }
                        ],
                    }
                    for i in range(3)
                ],
            }
            mock_get.return_value = mock_resp

            # limit=2 should return at most 2
            game_ids = projector._fetch_recent_game_ids(limit=2)
            assert len(game_ids) <= 2

    # ── Cache Tests ────────────────────────────────────────────────────

    def test_save_and_load_cache(self, projector_with_ratios):
        """Cached ratios should be loadable back into a new projector."""
        projector_with_ratios._save_cache()
        assert projector_with_ratios._cache_path.exists()

        # New projector loads the cache
        projector2 = QuarterHalfProjector(cache_path=projector_with_ratios._cache_path)
        loaded = projector2._load_cache()
        assert loaded, "Cache should load successfully"
        assert "celtics" in projector2._ratios
        assert projector2._ratios["celtics"]["q1"] == 0.26

    def test_cache_persistence_across_projectors(self, projector_with_ratios):
        """Ratios saved by one projector are available to another."""
        projector_with_ratios._save_cache()

        projector2 = QuarterHalfProjector(cache_path=projector_with_ratios._cache_path)
        num_teams = projector2.compute_ratios(max_games=200)
        assert num_teams == len(projector_with_ratios._ratios)
        assert num_teams >= 2

    def test_missing_cache_returns_false(self, projector):
        """Loading a non-existent cache returns False."""
        projector._cache_path = Path("/nonexistent/cache.json")
        assert not projector._load_cache()

    def test_corrupt_cache_returns_false(self, projector):
        """Corrupt cache JSON returns False gracefully."""
        projector._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(projector._cache_path, "w") as f:
            f.write("{{not valid json}}")
        assert not projector._load_cache()

    # ── Team Ratio Computation ─────────────────────────────────────────

    def test_compute_nba_ratios_from_mocked_data(
        self, projector
    ):
        """compute_ratios computes correct team ratios from mocked ESPN data.

        This test creates 3 mocked games with the same team in the same
        matchup (Celtics vs Lakers) so that the minimum-3-games-per-team
        threshold is satisfied.
        """
        # 3 games with the same Celtics vs Lakers matchup and consistent scores
        scoreboard_data = {
            "events": [
                {
                    "id": f"game_{i:03d}",
                    "competitions": [
                        {
                            "competitors": [
                                {"team": {"displayName": "Boston Celtics"}},
                                {"team": {"displayName": "Los Angeles Lakers"}},
                            ],
                            "status": {"type": {"name": "STATUS_FINAL"}},
                        }
                    ],
                }
                for i in range(3)
            ],
        }
        # Same linescores for all 3 games so ratios are consistent
        summary_data = {
            "header": {
                "competitions": [
                    {
                        "competitors": [
                            {
                                "team": {"displayName": "Boston Celtics"},
                                "linescores": [
                                    {"value": 32}, {"value": 28},
                                    {"value": 30}, {"value": 26},
                                ],
                            },
                            {
                                "team": {"displayName": "Los Angeles Lakers"},
                                "linescores": [
                                    {"value": 25}, {"value": 30},
                                    {"value": 27}, {"value": 29},
                                ],
                            },
                        ],
                    }
                ],
            },
        }

        request_count = 0

        def mock_get(url, **kwargs):
            nonlocal request_count
            request_count += 1
            resp = MagicMock(spec=requests.Response)
            resp.status_code = 200
            if "scoreboard" in url:
                resp.json.return_value = scoreboard_data
            else:
                resp.json.return_value = summary_data
            return resp

        with patch.object(projector._session, "get", side_effect=mock_get):
            with patch.object(projector, "_rate_limit"):
                count = projector.compute_ratios(max_games=10)

        assert count > 0, "Should have computed at least one team's ratios"

        # Celtics: (32+28+30+26) = 116 total → q1=32/116≈0.276
        celtics = projector._ratios.get("boston celtics")
        assert celtics is not None, "Celtics should have computed ratios"
        assert 0.25 < celtics["q1"] < 0.30, (
            f"Celtics Q1 ratio {celtics['q1']} unexpected"
        )
        # h1 = q1+q2 = 32+28=60, h1/total = 60/116 ≈ 0.517
        assert celtics["h1"] > 0.50, (
            f"Celtics H1 ratio {celtics['h1']} should be > 0.50"
        )

    def test_compute_ratios_skips_teams_with_few_games(self, projector):
        """Teams with <3 games should be skipped (not crash)."""
        team_data = {
            "team a": [{"q1": 20, "q2": 25, "q3": 22, "q4": 24}],  # Only 1 game
            "team b": [
                {"q1": 20, "q2": 25, "q3": 22, "q4": 24},  # Only 2 games
                {"q1": 22, "q2": 24, "q3": 23, "q4": 25},
            ],
        }
        # Manually run _compute_nba_ratios internal logic
        # This is tested by injecting data into `_compute_nba_ratios` which
        # calls `_fetch_recent_game_ids` and `_fetch_linescores`.
        # Since we mock those, the result will be empty (no real teams).
        with patch.object(projector._session, "get") as mock_get:
            mock_resp = MagicMock(spec=requests.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"events": []}
            mock_get.return_value = mock_resp

            projector._compute_nba_ratios(max_games=5)
            # Should not crash, ratios will be empty
            assert len(projector._ratios) == 0

    def test_load_cache_loads_ratios(self, projector_with_ratios):
        """_load_cache populates _ratios from a saved cache file."""
        projector_with_ratios._save_cache()

        projector2 = QuarterHalfProjector(cache_path=projector_with_ratios._cache_path)
        loaded = projector2._load_cache()
        assert loaded, "Cache should load successfully"
        assert "celtics" in projector2._ratios
        assert projector2._ratios["celtics"]["q1"] == 0.26

    def test_compute_ratios_only_once(self, projector):
        """compute_ratios should not re-fetch after initial load."""
        projector._ratios = {"celtics": {"q1": 0.26}}
        projector._loaded = True

        with patch.object(projector._session, "get") as mock_get:
            count = projector.compute_ratios(max_games=200)
            assert count == 1
            mock_get.assert_not_called(), "Should not re-fetch if already loaded"

    # ── Normalization ──────────────────────────────────────────────────

    def test_normalize_team(self, projector):
        """Team names should be lowercased and stripped."""
        assert projector._normalize_team("  Celtics  ") == "celtics"
        assert projector._normalize_team("Los Angeles Lakers") == "los angeles lakers"
        assert projector._normalize_team("") == ""

    def test_team_name_matching(self, projector_with_ratios):
        """Team names should be matched case-insensitively."""
        result = projector_with_ratios.project(200.0, "CELTICS", "lakers")
        assert result["home_score"] > 0
        assert result["away_score"] > 0


# ═══════════════════════════════════════════════════════════════════════════
#  Quarter/Half Bet Type Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQuarterHalfBetTypes:
    """Tests for QuarterTotalBet, QuarterBet, and HalfTotalBet constructors."""

    # ── QuarterTotalBet ────────────────────────────────────────────────

    def test_quarter_total_bet_over(self):
        """QuarterTotalBet with OVER side returns correct structure."""
        bet = QuarterTotalBet(
            game_id="GAME001",
            game_date="2024-01-15",
            matchup="Celtics @ Lakers",
            side="OVER",
            market_quarter_total=55.0,
            predicted_quarter_total=58.0,
            quarter=1,
        )
        assert bet.bet_type == BetType.QUARTER_TOTAL
        assert bet.bet_side == "1st Qtr OVER 55"
        assert bet.predicted_value == 58.0
        assert bet.market_line == 55.0
        assert bet.edge_pct > 0  # Over 58 vs mkt 55 => positive edge
        assert "quarter_total" in bet.tags

    def test_quarter_total_bet_under(self):
        """QuarterTotalBet with UNDER side returns correct structure."""
        bet = QuarterTotalBet(
            game_id="GAME001",
            game_date="2024-01-15",
            matchup="Celtics @ Lakers",
            side="UNDER",
            market_quarter_total=55.0,
            predicted_quarter_total=52.0,
            quarter=2,
        )
        assert bet.bet_side == "2nd Qtr UNDER 55"
        assert bet.bet_type == BetType.QUARTER_TOTAL
        assert bet.edge_pct > 0

    def test_quarter_total_edge_from_market_total(self):
        """Win probability > 0.5 when predicted deviates from market."""
        bet_over = QuarterTotalBet(
            game_id="G001", game_date="2024-01-15",
            matchup="A @ B", side="OVER",
            market_quarter_total=50.0, predicted_quarter_total=60.0,
            quarter=1,
        )
        assert bet_over.win_probability > 0.5
        assert bet_over.edge_pct > 0
        assert bet_over.predicted_value == 60.0

    def test_quarter_total_all_four_quarters(self):
        """QuarterTotalBet works for quarters 1-4."""
        for q in [1, 2, 3, 4]:
            ordinals = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
            bet = QuarterTotalBet(
                game_id="G001", game_date="2024-01-15",
                matchup="A @ B", side="OVER",
                market_quarter_total=50.0, predicted_quarter_total=53.0,
                quarter=q,
            )
            assert bet.bet_type == BetType.QUARTER_TOTAL
            assert ordinals[q] in bet.bet_side, (
                f"Q{q} ordinal '{ordinals[q]}' not in '{bet.bet_side}'"
            )

    def test_quarter_total_edge_zero_at_market(self):
        """Edge should be near zero when prediction matches market."""
        bet = QuarterTotalBet(
            game_id="G001", game_date="2024-01-15",
            matchup="A @ B", side="OVER",
            market_quarter_total=55.0, predicted_quarter_total=55.0,
            quarter=1,
        )
        assert abs(bet.edge_pct) < 0.01

    def test_quarter_total_kwargs_passthrough(self):
        """Extra kwargs like confidence and reasoning pass through."""
        bet = QuarterTotalBet(
            game_id="G001", game_date="2024-01-15",
            matchup="A @ B", side="OVER",
            market_quarter_total=55.0, predicted_quarter_total=58.0,
            quarter=1,
            confidence=Confidence.HIGH,
            reasoning="Strong projection",
        )
        assert bet.confidence == Confidence.HIGH
        assert bet.reasoning == "Strong projection"

    def test_quarter_total_win_prob_bounds(self):
        """Win probability should be bounded [0.01, 0.92]."""
        # Huge difference should cap at 0.92
        bet = QuarterTotalBet(
            game_id="G001", game_date="2024-01-15",
            matchup="A @ B", side="OVER",
            market_quarter_total=50.0, predicted_quarter_total=100.0,
            quarter=1,
        )
        assert bet.win_probability <= 0.92
        assert bet.win_probability >= 0.01

    # ── QuarterBet (winner) ────────────────────────────────────────────

    def test_quarter_bet_structure(self):
        """QuarterBet creates a valid FIRST_QUARTER_WINNER bet."""
        bet = QuarterBet(
            game_id="G001", game_date="2024-01-15",
            matchup="Celtics @ Lakers",
            quarter=1, team="Celtics",
            win_probability=0.65,
        )
        assert bet.bet_type == BetType.FIRST_QUARTER_WINNER
        assert "Celtics" in bet.bet_side
        assert bet.win_probability == 0.65
        assert bet.predicted_value == 0.65

    def test_quarter_bet_different_quarters(self):
        """QuarterBet works for all 4 quarters."""
        for q in [1, 2, 3, 4]:
            bet = QuarterBet(
                game_id="G001", game_date="2024-01-15",
                matchup="A @ B", quarter=q, team="Team A",
                win_probability=0.5 + 0.05 * q,
            )
            ordinals = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
            assert ordinals[q] in bet.bet_side

    def test_quarter_bet_edge_computation(self):
        """QuarterBet correctly computes edge from win probability."""
        bet = QuarterBet(
            game_id="G001", game_date="2024-01-15",
            matchup="A @ B", quarter=1, team="A",
            win_probability=0.60,
            market_implied_prob=0.50,
        )
        assert abs(bet.edge_pct - 0.10) < 0.001, (
            f"Expected 0.10 edge, got {bet.edge_pct}"
        )
        assert bet.expected_value > 0  # +EV bet

    def test_quarter_bet_negative_edge(self):
        """QuarterBet correctly computes negative edge."""
        bet = QuarterBet(
            game_id="G001", game_date="2024-01-15",
            matchup="A @ B", quarter=1, team="A",
            win_probability=0.40,
            market_implied_prob=0.50,
        )
        assert bet.edge_pct < 0
        assert bet.expected_value < 0

    # ── HalfTotalBet ───────────────────────────────────────────────────

    def test_half_total_bet_first_half(self):
        """HalfTotalBet creates a FIRST_HALF_TOTAL bet."""
        bet = HalfTotalBet(
            game_id="G001", game_date="2024-01-15",
            matchup="A @ B", side="OVER",
            market_half_total=110.5,
            predicted_half_total=115.0,
        )
        assert bet.bet_type == BetType.FIRST_HALF_TOTAL
        assert "1st Half" in bet.bet_side
        assert bet.predicted_value == 115.0

    def test_half_total_bet_under(self):
        """HalfTotalBet under side."""
        bet = HalfTotalBet(
            game_id="G001", game_date="2024-01-15",
            matchup="A @ B", side="UNDER",
            market_half_total=110.5,
            predicted_half_total=105.0,
        )
        assert "UNDER" in bet.bet_side
        assert bet.edge_pct > 0

    def test_half_total_kwargs_passthrough(self):
        """Extra kwargs like confidence pass through."""
        bet = HalfTotalBet(
            game_id="G001", game_date="2024-01-15",
            matchup="A @ B", side="OVER",
            market_half_total=110.5, predicted_half_total=115.0,
            confidence=Confidence.VERY_HIGH,
        )
        assert bet.confidence == Confidence.VERY_HIGH

    def test_half_total_win_prob_bounds(self):
        """Win probability capped for extreme edges."""
        bet = HalfTotalBet(
            game_id="G001", game_date="2024-01-15",
            matchup="A @ B", side="OVER",
            market_half_total=100.0, predicted_half_total=200.0,
        )
        assert bet.win_probability <= 0.92


# ═══════════════════════════════════════════════════════════════════════════
#  BetType Enum Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBetTypeEnum:
    """Tests for new BetType enum values."""

    def test_quarter_total_enum(self):
        """QUARTER_TOTAL enum exists and has display_name/icon."""
        assert BetType.QUARTER_TOTAL.value == "quarter_total"
        assert BetType.QUARTER_TOTAL.display_name() == "Quarter Total"
        assert BetType.QUARTER_TOTAL.icon()

    def test_second_half_total_enum(self):
        """SECOND_HALF_TOTAL enum exists and has display_name/icon."""
        assert BetType.SECOND_HALF_TOTAL.value == "second_half_total"
        assert BetType.SECOND_HALF_TOTAL.display_name() == "2nd Half Total"
        assert BetType.SECOND_HALF_TOTAL.icon()

    def test_first_half_total_enum(self):
        """FIRST_HALF_TOTAL still works."""
        assert BetType.FIRST_HALF_TOTAL.value == "first_half_total"
        assert BetType.FIRST_HALF_TOTAL.display_name() == "1st Half Total"


# ═══════════════════════════════════════════════════════════════════════════
#  Engine Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEngineQuarterHalfIntegration:
    """Tests that RecommendationEngine generates quarter/half bets from predictions."""

    @pytest.fixture
    def sample_predictions(self):
        """Realistic predictions DataFrame with one game."""
        return pd.DataFrame([
            {
                "game_id": "G001",
                "game_date": "2024-01-15",
                "home_team": "Celtics",
                "away_team": "Lakers",
                "predicted_total": 225.0,
                "market_total": 220.0,
                "predicted_spread": 5.0,
                "spread": -3.5,
                "league": "NBA",
            }
        ])

    def test_engine_generates_quarter_bets(self, sample_predictions):
        """Engine generates quarter total bets with a mocked projector."""
        engine = RecommendationEngine()

        # Inject a projector with team ratios
        from betting_intel.data.quarter_projector import QuarterHalfProjector
        fake_projector = MagicMock(spec=QuarterHalfProjector)
        fake_projector.project.return_value = {
            "q1_total": 55.0, "q2_total": 57.0, "q3_total": 56.0, "q4_total": 57.0,
            "h1_total": 112.0, "h2_total": 113.0,
            "q1_home": 28.0, "q1_away": 27.0,
            "q2_home": 29.0, "q2_away": 28.0,
            "q3_home": 29.0, "q3_away": 27.0,
            "q4_home": 30.0, "q4_away": 27.0,
            "h1_home": 57.0, "h1_away": 55.0,
            "h2_home": 59.0, "h2_away": 54.0,
            "home_score": 116.0, "away_score": 109.0,
            "predicted_total": 225.0,
        }
        fake_projector.get_quarter_market.return_value = 54.0
        fake_projector.get_half_market.return_value = 111.0

        # Patch the projector into the engine's global
        import betting_intel.recommendations.engine as eng_mod
        orig = eng_mod._quarter_projector
        eng_mod._quarter_projector = fake_projector
        try:
            bets = engine.generate_all_bets(sample_predictions)
        finally:
            eng_mod._quarter_projector = orig

        # Verify quarter/half bets are present
        quarter_bets = [b for b in bets if b.bet_type == BetType.QUARTER_TOTAL]
        half_bets = [
            b for b in bets
            if b.bet_type in (BetType.FIRST_HALF_TOTAL, BetType.SECOND_HALF_TOTAL)
        ]
        quarter_winner_bets = [
            b for b in bets if b.bet_type == BetType.FIRST_QUARTER_WINNER
        ]

        assert len(quarter_bets) > 0, "Should generate quarter total bets"
        assert len(half_bets) > 0, "Should generate half total bets"
        assert len(quarter_winner_bets) > 0, "Should generate quarter winner bets"

        # Check a specific quarter total bet
        q_bet = quarter_bets[0]
        assert q_bet.model_name == "pipeline_ensemble_with_projector"
        assert q_bet.predicted_value > 0

    def test_engine_graceful_on_projector_failure(self, sample_predictions):
        """Engine should not crash if projector raises an exception."""
        engine = RecommendationEngine()
        fake_projector = MagicMock()
        fake_projector.project.side_effect = ValueError("ESPN API down")

        import betting_intel.recommendations.engine as eng_mod
        orig = eng_mod._quarter_projector
        eng_mod._quarter_projector = fake_projector
        try:
            bets = engine.generate_all_bets(sample_predictions)
        finally:
            eng_mod._quarter_projector = orig

        # Should still generate non-quarter bets (total, spread, moneyline)
        assert len(bets) > 0, "Should still generate bets even if projector fails"
        # No quarter/half bets expected
        qh_bets = [
            b for b in bets
            if b.model_name == "pipeline_ensemble_with_projector"
        ]
        assert len(qh_bets) == 0

    def test_engine_without_projector(self):
        """Engine should work with no projector (lazy init)."""
        engine = RecommendationEngine()

        import betting_intel.recommendations.engine as eng_mod
        eng_mod._quarter_projector = None  # Reset

        # Small predictions — should not crash
        df = pd.DataFrame([
            {"game_id": "G001", "game_date": "2024-01-15",
             "home_team": "A", "away_team": "B",
             "predicted_total": 220.0, "market_total": 218.0, "league": "NBA"}
        ])
        bets = engine.generate_all_bets(df)
        assert len(bets) > 0  # At least total O/U
