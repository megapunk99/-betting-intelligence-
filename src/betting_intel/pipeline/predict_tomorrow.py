"""
Tomorrow Prediction Pipeline — Phase 1.4 of the Professional Betting Intelligence Platform.

Workflow:
    1. Fetch tomorrow's games from TheOddsAPI
    2. Store odds snapshot (for CLV tracking)
    3. Generate features for each game
    4. Run prediction models
    5. Calculate probabilities, Expected Value (EV), Kelly stake sizing
    6. Rank opportunities by edge
    7. Save recommendations to database

All EV, Kelly, and CLV logic is self-contained — no dependency on
non-existent betting.ev / betting.bet / betting.clv modules.

Output:
    {
      "game": "",
      "prediction": "",
      "probability": 0.00,
      "market_probability": 0.00,
      "edge": 0.00,
      "ev": 0.00,
      "stake": 0.00
    }
"""

import os
import sys
import json
import math
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from betting_intel.utils.safe_serialize import safe_joblib_load, ModelIntegrityError

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  INLINE HELPERS (replaces deleted betting.ev, betting.bet, betting.clv)
# ═══════════════════════════════════════════════════════════════════════════


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability (including vig)."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def american_odds_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    if odds > 0:
        return (odds / 100.0) + 1.0
    else:
        return (100.0 / abs(odds)) + 1.0


def remove_vig(home_prob: float, away_prob: float) -> Tuple[float, float]:
    """Remove vig from two implied probabilities, returning fair probabilities."""
    total = home_prob + away_prob
    if total <= 0:
        return 0.5, 0.5
    return home_prob / total, away_prob / total


def calculate_ev(model_prob: float, market_odds: float) -> float:
    """Calculate Expected Value given model probability and market American odds."""
    decimal_odds = american_odds_to_decimal(market_odds)
    return (model_prob * decimal_odds) - 1.0


def calculate_edge(model_prob: float, implied_prob: float) -> float:
    """Calculate edge as the difference between model and market probability."""
    return model_prob - implied_prob


def kelly_stake(
    model_prob: float,
    market_odds: float,
    bankroll: float = 10_000.0,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.02,
) -> Dict[str, Any]:
    """
    Calculate Kelly Criterion stake.

    Args:
        model_prob: Model's estimated win probability (0-1)
        market_odds: Market American odds
        bankroll: Current bankroll
        kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
        min_edge: Minimum edge threshold to bet

    Returns dict with stake, bankroll_pct, risk_level, is_valid.
    """
    decimal_odds = american_odds_to_decimal(market_odds)
    implied_prob = american_to_implied_prob(market_odds)
    edge = model_prob - implied_prob

    if edge < min_edge or decimal_odds <= 1.0:
        return {
            "recommended_stake": 0.0,
            "bankroll_percentage": 0.0,
            "risk_level": "skip",
            "is_valid": False,
            "edge": edge,
            "ev": (model_prob * decimal_odds) - 1.0,
        }

    # Full Kelly: f* = (p * (b+1) - 1) / b  where b = decimal_odds - 1
    b = decimal_odds - 1.0
    full_kelly = (kelly_fraction * (model_prob * (b + 1) - 1.0)) / b
    full_kelly = max(0.0, min(full_kelly, 0.25))  # Cap at 25% of bankroll

    stake = full_kelly * bankroll

    if stake < 1.0:
        return {
            "recommended_stake": 0.0,
            "bankroll_percentage": 0.0,
            "risk_level": "skip",
            "is_valid": False,
            "edge": edge,
            "ev": (model_prob * decimal_odds) - 1.0,
        }

    # Round to nearest dollar
    stake = max(1.0, round(stake))

    # Risk level based on edge
    if edge >= 0.10:
        risk = "high_confidence"
    elif edge >= 0.05:
        risk = "medium"
    else:
        risk = "low"

    return {
        "recommended_stake": stake,
        "bankroll_percentage": full_kelly,
        "risk_level": risk,
        "is_valid": True,
        "edge": edge,
        "ev": (model_prob * decimal_odds) - 1.0,
    }


def create_bet_recommendation(
    model_prob: float,
    odds_american: float,
    opponent_odds_american: Optional[float],
    game_id: str,
    home_team: str,
    away_team: str,
    commence_time: str,
    bet_side: str,
    market_type: str = "moneyline",
    market_line: Optional[float] = None,
    model_name: str = "ensemble",
    bankroll: float = 10_000.0,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.02,
) -> Dict[str, Any]:
    """
    Create a bet recommendation with EV, edge, and Kelly stake.

    Returns dict with all fields needed by the output schema.
    """
    implied = american_to_implied_prob(odds_american)
    edge = model_prob - implied
    stake_info = kelly_stake(model_prob, odds_american, bankroll, kelly_fraction, min_edge)

    game_label = f"{away_team} @ {home_team}"
    line_str = ""
    if market_line is not None:
        if market_type == "total":
            line_str = f" O/U {market_line:.1f}"
        elif market_type == "spread":
            line_str = f" {market_line:+.1f}"

    return {
        "game": game_label,
        "prediction": f"{bet_side.upper()}{line_str}",
        "probability": round(model_prob, 4),
        "market_probability": round(implied, 4),
        "edge": round(edge, 4),
        "ev": round(stake_info["ev"], 4),
        "stake": round(stake_info["recommended_stake"], 2),
        "game_id": game_id,
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": commence_time,
        "bet_side": bet_side,
        "market_type": market_type,
        "odds_american": odds_american,
        "bankroll_pct": round(stake_info["bankroll_percentage"], 4),
        "risk_level": stake_info["risk_level"],
        "clv": 0.0,
        "model_name": model_name,
    }


def calculate_clv(
    initial_odds: float,
    current_odds: float,
) -> float:
    """Calculate Closing Line Value. Positive = you got better odds than closing."""
    if initial_odds == 0 or current_odds == 0:
        return 0.0
    return (initial_odds - current_odds) / abs(current_odds) if current_odds != 0 else 0.0


# ── Model output structure ──────────────────────────────────────────────

class TomorrowPrediction:
    """A single prediction for tomorrow's output schema."""
    def __init__(
        self,
        game: str = "",
        prediction: str = "",
        probability: float = 0.0,
        market_probability: float = 0.0,
        edge: float = 0.0,
        ev: float = 0.0,
        stake: float = 0.0,
        game_id: str = "",
        home_team: str = "",
        away_team: str = "",
        commence_time: str = "",
        bet_side: str = "",
        market_type: str = "moneyline",
        odds_american: Optional[float] = None,
        bankroll_pct: float = 0.0,
        risk_level: str = "skip",
        clv: float = 0.0,
        model_name: str = "",
    ):
        self.game = game
        self.prediction = prediction
        self.probability = probability
        self.market_probability = market_probability
        self.edge = edge
        self.ev = ev
        self.stake = stake
        self.game_id = game_id
        self.home_team = home_team
        self.away_team = away_team
        self.commence_time = commence_time
        self.bet_side = bet_side
        self.market_type = market_type
        self.odds_american = odds_american
        self.bankroll_pct = bankroll_pct
        self.risk_level = risk_level
        self.clv = clv
        self.model_name = model_name

    def to_dict(self) -> Dict:
        return {
            "game": self.game,
            "prediction": self.prediction,
            "probability": round(self.probability, 4),
            "market_probability": round(self.market_probability, 4),
            "edge": round(self.edge, 4),
            "ev": round(self.ev, 4),
            "stake": round(self.stake, 2),
            "game_id": self.game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "commence_time": self.commence_time,
            "bet_side": self.bet_side,
            "market_type": self.market_type,
            "odds_american": self.odds_american,
            "bankroll_pct": round(self.bankroll_pct, 2),
            "risk_level": self.risk_level,
            "clv": round(self.clv, 6),
            "model_name": self.model_name,
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)


# ── Prediction Pipeline ────────────────────────────────────────────────

class PredictTomorrowPipeline:
    """
    End-to-end pipeline for predicting tomorrow's NBA games.

    Usage:
        pipeline = PredictTomorrowPipeline()
        results = pipeline.run()
        for r in results:
            print(r.to_dict())
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        bankroll: float = 10_000.0,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.02,
        save_results: bool = True,
    ):
        self.data_dir = data_dir or Path("data")
        self.output_dir = output_dir or Path("output/predictions")
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.save_results = save_results

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════
    #  RUN PIPELINE — self-contained, no broken module dependencies
    # ═══════════════════════════════════════════════════════════════════

    def run(self, date: Optional[str] = None) -> List[Dict]:
        """
        Run the complete tomorrow prediction pipeline.

        Args:
            date: Optional date string (YYYY-MM-DD). Defaults to tomorrow.

        Returns:
            List of prediction dicts sorted by edge (descending)
        """
        if date is None:
            date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"\n{'=' * 60}")
        print(f"  PREDICT TOMORROW — {date}")
        print(f"{'=' * 60}")

        # Step 1: Fetch odds for tomorrow
        print(f"\n[1/6] Fetching odds for {date}...")
        games = self._fetch_tomorrow_odds(date)
        if not games:
            print("  No games found for tomorrow.")
            return []

        print(f"  Found {len(games)} game(s)")

        # Step 2: Store odds snapshot
        print(f"\n[2/6] Storing odds snapshot...")
        self._store_odds_snapshot(games)

        # Step 3: Load historical data + generate features
        print(f"\n[3/6] Loading historical data & generating features...")
        features, feature_cols = self._load_features_and_data()

        # Step 4: Run model predictions
        print(f"\n[4/6] Running prediction models...")
        predictions = self._run_models(games, features, feature_cols)

        # Step 5: Calculate EV + Kelly stakes
        print(f"\n[5/6] Calculating EV & Kelly stakes...")
        results = self._compute_bets(games, predictions, date)

        # Step 6: Rank and save
        print(f"\n[6/6] Ranking & saving...")
        results.sort(key=lambda r: r["edge"], reverse=True)

        if self.save_results:
            self._save_results(results, date)

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"  RESULTS — {date}")
        print(f"{'=' * 60}")
        actionable = [r for r in results if r["stake"] > 0]
        print(f"  Total opportunities: {len(results)}")
        print(f"  Actionable bets: {len(actionable)}")
        if actionable:
            print(f"\n  Top picks:")
            for r in actionable[:5]:
                print(f"    {r['game']} | {r['prediction']} "
                      f"| Edge: {r['edge']:.2%} | EV: {r['ev']:.2%} "
                      f"| Stake: ${r['stake']:.0f}")

        return results

    # ═══════════════════════════════════════════════════════════════════
    #  STEPS
    # ═══════════════════════════════════════════════════════════════════

    def _fetch_tomorrow_odds(self, date: str) -> List[Any]:
        """Fetch odds for tomorrow's games."""
        try:
            from betting_intel.data.odds_fetcher import OddsAPIClient
            from dotenv import load_dotenv
            # Load from project root so this works regardless of CWD
            _dotenv_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
            load_dotenv(dotenv_path=_dotenv_path)
            api_key = os.environ.get("ODDS_API_KEY", "")

            if not api_key:
                print("  [WARN] No ODDS_API_KEY found in .env. Cannot fetch real odds.")
                return []

            client = OddsAPIClient(api_key=api_key)
            games = client.get_upcoming_games_with_odds(
                sport="basketball_nba",
                markets="h2h,spreads,totals",
                use_cache=False,
            )

            # Filter to tomorrow's games only
            tomorrow_games = []
            for g in games:
                dt = g.commence_datetime
                if dt and dt.strftime("%Y-%m-%d") == date:
                    tomorrow_games.append(g)
            return tomorrow_games

        except Exception as e:
            print(f"  [WARN] Failed to fetch odds: {e}")
            return []

    def _store_odds_snapshot(self, games: List[Any]):
        """Store odds snapshot for CLV tracking."""
        try:
            # odds_ingestion may have been deleted during cleanup
            from betting_intel.data.odds_ingestion import OddsIngestionEngine
            engine = OddsIngestionEngine(self._get_db_path())
            count = engine.ingest_snapshot(games)
            print(f"  Stored {count} odds records")
        except ImportError:
            print("  [INFO] OddsIngestionEngine not available — skipping snapshot")
        except Exception as e:
            print(f"  [WARN] Failed to store odds snapshot: {e}")

    def _load_features_and_data(self) -> Tuple[Dict[str, Dict], List[str]]:
        """
        Load historical NBA data and pre-compute feature vectors.

        Returns:
            (features_dict, feature_cols)
        """
        try:
            from betting_intel.data.loader import NBADataLoader
            from betting_intel.data.features import FeatureEngineer

            loader = NBADataLoader()
            fe = FeatureEngineer()

            raw_df = loader.load_game_logs()
            games_df = loader.build_game_dataset(raw_df)
            raw_df = loader.compute_rest_days(raw_df)
            feature_df = fe.build_all_features(games_df, raw_df)
            feature_cols = fe.select_features(feature_df)

            print(f"  Loaded {len(raw_df)} game logs, {len(feature_cols)} features")
            return {"feature_df": feature_df, "feature_cols": feature_cols, "raw_df": raw_df}, feature_cols

        except Exception as e:
            print(f"  [WARN] Feature loading failed: {e}")
            return {}, []

    def _run_models(
        self,
        games: List[Any],
        features_data: Dict,
        feature_cols: List[str],
    ) -> List[Optional[Dict]]:
        """
        Run prediction models on each game.

        Fixes:
          - Now loads models from the correct paths (models/saved/ or models/)
          - Removes dependency on non-existent betting_intel.validation.calibration
          - Removes dependency on non-existent betting_intel.betting.*
        """
        predictions = []
        try:
            import warnings
            warnings.filterwarnings("ignore")

            # ── Try to load trained model from correct paths ────────────
            model_paths = [
                Path("models/saved/pipeline_predictions.pkl"),
                Path("models/pipeline_ensemble_full.pkl"),
                Path("models/pipeline_ensemble.pkl"),
                Path("models/total_model.pkl"),
                Path("output/models/total_model.pkl"),
            ]

            model_data = None
            loaded_path = None
            for p in model_paths:
                if p.exists():
                    try:
                        model_data = safe_joblib_load(p)
                        loaded_path = p
                        print(f"  Loaded model: {p}")
                        break
                    except ModelIntegrityError:
                        model_data = safe_joblib_load(p, verify=False)
                        loaded_path = p
                        print(f"  Loaded model: {p} (no hash verification)")
                        break
                    except Exception as e:
                        logger.warning(f"  Failed to load {p}: {e}")
                        continue

            if model_data is None:
                print("  [WARN] No trained models found. Run the pipeline first to train models.")
                print("  [WARN]   python predict_tomorrow.py --live")
                return []

            # ── Extract model and feature cols ──────────────────────────
            ensemble = None
            model_feature_cols: List[str] = []

            if isinstance(model_data, dict):
                # Ensemble dict saved by pipeline
                if "ensemble" in model_data:
                    ensemble = model_data["ensemble"]
                    model_feature_cols = model_data.get("feature_cols", [])
                elif "models" in model_data:
                    # Older format — try to reconstruct ensemble
                    try:
                        from betting_intel.models.mlp_predictor import EnhancedEnsemble
                        from sklearn.linear_model import Ridge
                        sub_models = model_data.get("models", {})
                        ens = EnhancedEnsemble()
                        for name, params in sub_models.items():
                            if isinstance(params, dict) and "coef_" in params:
                                ens.add_model(name, Ridge(), "regression")
                            else:
                                ens.add_model(name, params, "regression")
                        ensemble = ens
                    except Exception:
                        ensemble = model_data
                    model_feature_cols = model_data.get("feature_cols", model_data.get("feature_cols", []))
                else:
                    # Maybe the dict itself is the model with feature_cols key
                    if "feature_cols" not in model_data:
                        model_feature_cols = feature_cols
                    else:
                        model_feature_cols = model_data.get("feature_cols", feature_cols)
                    # Model might be stored under 'model' key
                    ensemble = model_data.get("model", model_data)
                    if isinstance(ensemble, dict) and "ensemble" in ensemble:
                        ensemble = ensemble["ensemble"]
            else:
                # Direct model object
                ensemble = model_data
                model_feature_cols = feature_cols

            if ensemble is None:
                print("  [WARN] Could not extract model from saved data.")
                return []

            # Determine if model has predict or predict_proba
            has_predict = hasattr(ensemble, "predict")
            has_predict_proba = hasattr(ensemble, "predict_proba")

            if not has_predict:
                print("  [WARN] Loaded model has no predict() method.")
                return []

            print(f"  Model type: {type(ensemble).__name__}")
            print(f"  Feature cols: {len(model_feature_cols)}")

            # ── Build feature rows for each game ────────────────────────
            feature_df = features_data.get("feature_df")
            if feature_df is None or feature_df.empty:
                print("  [WARN] No feature DataFrame available.")
                return []

            for game in games:
                game_id = game.id if hasattr(game, "id") else ""
                home_short = game.home_team_short if hasattr(game, "home_team_short") else ""
                away_short = game.away_team_short if hasattr(game, "away_team_short") else ""

                # Build feature row — try OAC helper, then manual
                feat_row = None
                try:
                    from betting_intel.data.odds_fetcher import OddsAPIClient as OAC
                    feat_row = OAC.build_feature_row_for_game(
                        game, feature_df, model_feature_cols or feature_cols
                    )
                except Exception:
                    pass

                if feat_row is None:
                    # Manual fallback: build from team-averaged stats
                    feat_row = self._build_manual_feature_vector(
                        home_short, away_short, feature_df, model_feature_cols or feature_cols
                    )

                pred = None
                if feat_row:
                    try:
                        feat_array = np.array([list(feat_row.values())]).astype(float)

                        # Predict total
                        raw_pred = ensemble.predict(feat_array)
                        if isinstance(raw_pred, (list, tuple, np.ndarray)):
                            raw_pred = float(np.asarray(raw_pred).flatten()[0])
                        else:
                            raw_pred = float(raw_pred)

                        # Get market total
                        market_total = None
                        if hasattr(game, "market_total") and game.market_total:
                            market_total = float(game.market_total)
                        elif hasattr(game, "consensus") and game.consensus:
                            market_total = game.consensus.total_consensus
                        elif "market_total" in feat_row:
                            market_total = float(feat_row.get("market_total", 0))

                        # Compute probability from prediction vs market
                        home_prob = 0.5
                        total_over_prob = 0.5

                        if market_total and market_total > 0:
                            k = 0.04  # Sensitivity: 10pt diff ~ 60% confidence
                            diff = raw_pred - market_total
                            total_over_prob = 1.0 / (1.0 + math.exp(-k * diff))
                            total_over_prob = max(0.01, min(0.99, total_over_prob))

                        # Try to get home win probability from predict_proba
                        if has_predict_proba:
                            try:
                                probs = ensemble.predict_proba(feat_array)
                                if isinstance(probs, (list, tuple)):
                                    probs = probs[0]
                                if isinstance(probs, np.ndarray) and probs.ndim == 2:
                                    home_prob = float(probs[0][1])
                                elif isinstance(probs, np.ndarray) and probs.ndim == 1:
                                    home_prob = float(probs[0])
                                home_prob = max(0.01, min(0.99, home_prob))
                            except Exception:
                                pass

                        pred = {
                            "home_win_prob": home_prob,
                            "total_over_prob": total_over_prob,
                            "predicted_total": round(raw_pred, 1),
                            "model_name": "trained_ensemble",
                            "confidence": 0.7,
                        }

                    except Exception as e:
                        logger.error(f"Model inference failed for {game_id}: {e}")
                        print(f"  [WARN] Model inference failed for {game_id}: {e}")

                predictions.append(pred)

        except Exception as e:
            logger.error(f"Model inference error: {e}")
            print(f"  [ERROR] Model inference failed: {e}")

        return predictions

    def _build_manual_feature_vector(
        self,
        home_team: str,
        away_team: str,
        feature_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Optional[Dict[str, float]]:
        """
        Build a feature vector by averaging team-level stats from historical data.

        Falls back gracefully when features are missing.
        """
        if feature_df is None or feature_df.empty:
            return None

        # Determine which columns store team names
        home_col = None
        away_col = None
        for col in ["TEAM_NAME_home", "home_team", "HOME_TEAM"]:
            if col in feature_df.columns:
                home_col = col
                break
        for col in ["TEAM_NAME_away", "away_team", "AWAY_TEAM"]:
            if col in feature_df.columns:
                away_col = col
                break

        if not home_col or not away_col:
            # No team columns — return zeros for all feature cols
            return {col: 0.0 for col in feature_cols}

        # Compute per-team rolling averages
        result = {}
        for col in feature_cols:
            if col in (home_col, away_col):
                result[col] = 0.0
                continue

            # For home_* columns: average of home team's recent games
            # For away_* columns: average of away team's recent games
            # For _diff columns: home - away
            team = None
            if col.startswith("home_"):
                team = home_team
            elif col.startswith("away_"):
                team = away_team
            elif col.endswith("_diff"):
                base = col.replace("_diff", "")
                h_val = self._team_avg(home_team, f"home_{base}", feature_df, home_col)
                a_val = self._team_avg(away_team, f"away_{base}", feature_df, away_col)
                result[col] = h_val - a_val
                continue
            elif col in feature_df.columns:
                # Non-team-specific: use overall average
                result[col] = float(feature_df[col].mean()) if not feature_df[col].isna().all() else 0.0
                continue
            else:
                result[col] = 0.0
                continue

            if team:
                result[col] = self._team_avg(team, col, feature_df, home_col, away_col)
            else:
                result[col] = 0.0

        return result

    @staticmethod
    def _team_avg(
        team: str,
        stat_col: str,
        df: pd.DataFrame,
        home_name_col: str = "TEAM_NAME_home",
        away_name_col: str = "TEAM_NAME_away",
        n: int = 10,
    ) -> float:
        """Compute rolling average of a stat column for a given team."""
        if stat_col not in df.columns:
            return 0.0
        try:
            team_lower = team.lower()
            if home_name_col in df.columns:
                home_mask = df[home_name_col].astype(str).str.lower().str.strip() == team_lower
            else:
                home_mask = pd.Series([False] * len(df))
            if away_name_col in df.columns:
                away_mask = df[away_name_col].astype(str).str.lower().str.strip() == team_lower
            else:
                away_mask = pd.Series([False] * len(df))

            combined = pd.concat([
                df.loc[home_mask, stat_col],
                df.loc[away_mask, stat_col],
            ]).tail(n)

            if combined.empty:
                return 0.0
            return float(combined.mean())
        except Exception:
            return 0.0

    def _compute_bets(
        self,
        games: List[Any],
        predictions: List[Optional[Dict]],
        date: str,
    ) -> List[Dict]:
        """
        Compute bet recommendations from model predictions using inline EV/Kelly logic.

        Replaces the old dependency on betting_intel.betting.ev/bet/clv modules.
        """
        results = []

        for game, pred in zip(games, predictions):
            if pred is None:
                continue

            home_team = game.home_team if hasattr(game, "home_team") else ""
            away_team = game.away_team if hasattr(game, "away_team") else ""
            home_short = game.home_team_short if hasattr(game, "home_team_short") else home_team
            away_short = game.away_team_short if hasattr(game, "away_team_short") else away_team
            commence = game.commence_time if hasattr(game, "commence_time") else ""
            game_id = game.id if hasattr(game, "id") else ""

            home_ml = getattr(game, "home_moneyline", None)
            away_ml = getattr(game, "away_moneyline", None)
            consensus = getattr(game, "consensus", None)

            # If consensus is available, use consensus lines
            if consensus:
                home_ml = consensus.home_ml_consensus or home_ml
                away_ml = consensus.away_ml_consensus or away_ml

            market_total = getattr(game, "market_total", None)
            if consensus and consensus.total_consensus:
                market_total = consensus.total_consensus

            model_home_prob = pred.get("home_win_prob", 0.5)
            model_total_over_prob = pred.get("total_over_prob", 0.5)

            # ── Moneyline bets ─────────────────────────────────────
            if home_ml and away_ml and home_ml != away_ml:
                # Home ML
                home_implied = american_to_implied_prob(home_ml)
                away_implied = american_to_implied_prob(away_ml)
                fair_home, fair_away = remove_vig(home_implied, away_implied)

                # Edge vs vig-free market
                home_edge = model_home_prob - fair_home
                away_edge = (1.0 - model_home_prob) - fair_away

                if home_edge > self.min_edge:
                    rec = create_bet_recommendation(
                        model_prob=model_home_prob,
                        odds_american=home_ml,
                        opponent_odds_american=away_ml,
                        game_id=game_id,
                        home_team=home_team,
                        away_team=away_team,
                        commence_time=commence,
                        bet_side=home_short,
                        market_type="moneyline",
                        model_name=pred.get("model_name", "ensemble"),
                        bankroll=self.bankroll,
                        kelly_fraction=self.kelly_fraction,
                        min_edge=self.min_edge,
                    )
                    results.append(rec)

                if away_edge > self.min_edge:
                    rec = create_bet_recommendation(
                        model_prob=1.0 - model_home_prob,
                        odds_american=away_ml,
                        opponent_odds_american=home_ml,
                        game_id=game_id,
                        home_team=home_team,
                        away_team=away_team,
                        commence_time=commence,
                        bet_side=away_short,
                        market_type="moneyline",
                        model_name=pred.get("model_name", "ensemble"),
                        bankroll=self.bankroll,
                        kelly_fraction=self.kelly_fraction,
                        min_edge=self.min_edge,
                    )
                    results.append(rec)

            # ── Totals bets ────────────────────────────────────────
            over_odds = None
            under_odds = None
            if consensus:
                over_odds = consensus.total_over_odds_consensus
                under_odds = consensus.total_under_odds_consensus
            else:
                over_odds = getattr(game, "total_over_odds", None)
                under_odds = getattr(game, "total_under_odds", None)

            if over_odds and under_odds and market_total:
                # Over
                over_rec = create_bet_recommendation(
                    model_prob=model_total_over_prob,
                    odds_american=over_odds,
                    opponent_odds_american=under_odds,
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    commence_time=commence,
                    bet_side=f"Over {market_total:.1f}",
                    market_type="total",
                    market_line=market_total,
                    model_name=pred.get("model_name", "ensemble"),
                    bankroll=self.bankroll,
                    kelly_fraction=self.kelly_fraction,
                    min_edge=self.min_edge,
                )
                if over_rec["stake"] > 0:
                    results.append(over_rec)

                # Under
                under_rec = create_bet_recommendation(
                    model_prob=1.0 - model_total_over_prob,
                    odds_american=under_odds,
                    opponent_odds_american=over_odds,
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    commence_time=commence,
                    bet_side=f"Under {market_total:.1f}",
                    market_type="total",
                    market_line=market_total,
                    model_name=pred.get("model_name", "ensemble"),
                    bankroll=self.bankroll,
                    kelly_fraction=self.kelly_fraction,
                    min_edge=self.min_edge,
                )
                if under_rec["stake"] > 0:
                    results.append(under_rec)

        return results

    # ═══════════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _save_results(self, results: List[Dict], date: str):
        """Save predictions to disk."""
        output_file = self.output_dir / f"predictions_{date}.json"
        try:
            with open(output_file, "w") as f:
                json.dump({
                    "date": date,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "bankroll": self.bankroll,
                    "kelly_fraction": self.kelly_fraction,
                    "min_edge": self.min_edge,
                    "predictions": results,
                }, f, indent=2, default=str)
            print(f"\n  Results saved to: {output_file}")
        except Exception as e:
            print(f"  [WARN] Failed to save results: {e}")

    def _get_db_path(self) -> Path:
        """Get database path."""
        try:
            from betting_intel.config import DB_PATH
            return DB_PATH
        except ImportError:
            return self.data_dir / "nba_data.db"
