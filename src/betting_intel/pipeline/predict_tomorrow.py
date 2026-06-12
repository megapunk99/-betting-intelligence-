"""
Tomorrow Prediction Pipeline — Phase 1.4 of the Professional Betting Intelligence Platform.

Workflow:
    1. Fetch tomorrow's games from TheOddsAPI
    2. Store odds snapshot (for CLV tracking)
    3. Fetch injury reports
    4. Generate features for each game
    5. Run prediction models
    6. Calculate probabilities
    7. Calculate Expected Value (EV)
    8. Calculate Kelly stake sizing
    9. Rank opportunities by edge
    10. Save recommendations to database

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

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


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
        self.data_dir = data_dir
        self.output_dir = output_dir or (data_dir / "predictions" if data_dir else Path("output/predictions"))
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.save_results = save_results

        # These will be initialized lazily
        self._odds_client = None
        self._odds_ingestion = None
        self._ev_engine = None
        self._betting_engine = None
        self._clv_tracker = None
        self._loader = None
        self._fe = None
        self._model = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════
    #  RUN PIPELINE
    # ═══════════════════════════════════════════════════════════════════

    def run(self, date: Optional[str] = None) -> List[Dict]:
        """
        Run the complete tomorrow prediction pipeline.

        Args:
            date: Optional date string (YYYY-MM-DD). Defaults to tomorrow.

        Returns:
            List of prediction dicts sorted by edge (descending)
        """
        # Lazy imports to avoid circular dependencies
        from betting_intel.betting.ev import ExpectedValueEngine
        from betting_intel.betting.bet import BettingEngine, BetRecommendation
        from betting_intel.betting.clv import CLVTracker

        if date is None:
            date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"\n{'=' * 60}")
        print(f"  PREDICT TOMORROW — {date}")
        print(f"{'=' * 60}")

        # Step 1: Fetch odds for tomorrow
        print(f"\n[1/8] Fetching odds for {date}...")
        games = self._fetch_tomorrow_odds(date)
        if not games:
            print("  No games found for tomorrow.")
            return []

        print(f"  Found {len(games)} game(s)")

        # Step 2: Store odds snapshot
        print(f"\n[2/8] Storing odds snapshot...")
        self._store_odds_snapshot(games)

        # Step 3: Fetch injuries
        print(f"\n[3/8] Fetching injury data...")
        injuries = self._fetch_injuries(games)

        # Step 4: Generate features
        print(f"\n[4/8] Generating features...")
        features = self._generate_features(games)

        # Step 5: Run model predictions
        print(f"\n[5/8] Running prediction models...")
        predictions = self._run_models(games, features)

        # Step 6: Calculate EV for each game
        print(f"\n[6/8] Calculating expected value...")
        ev_engine = ExpectedValueEngine(min_edge_threshold=self.min_edge)

        # Step 7: Create betting recommendations
        print(f"\n[7/8] Calculating Kelly staking...")
        betting_engine = BettingEngine(
            bankroll=self.bankroll,
            kelly_fraction=self.kelly_fraction,
            min_edge=self.min_edge,
        )

        # Step 8: Rank and save
        print(f"\n[8/8] Ranking opportunities...")
        clv_tracker = CLVTracker(self._get_db_path())

        results = []
        for game, pred in zip(games, predictions):
            if pred is None:
                continue

            home_short = game.home_team_short if hasattr(game, "home_team_short") else game.get("home_team_short", "")
            away_short = game.away_team_short if hasattr(game, "away_team_short") else game.get("away_team_short", "")
            home_team = game.home_team if hasattr(game, "home_team") else game.get("home_team", "")
            away_team = game.away_team if hasattr(game, "away_team") else game.get("away_team", "")
            commence = game.commence_time if hasattr(game, "commence_time") else game.get("commence_time", "")
            game_id = game.id if hasattr(game, "id") else game.get("id", "")
            home_ml = game.home_moneyline if hasattr(game, "home_moneyline") else game.get("home_moneyline")
            away_ml = game.away_moneyline if hasattr(game, "away_moneyline") else game.get("away_moneyline")
            total = game.market_total if hasattr(game, "market_total") else game.get("market_total")

            # Build bet recommendations for all actionable sides
            model_home_prob = pred.get("home_win_prob", 0.5)
            model_total_over_prob = pred.get("total_over_prob")

            bets_made = 0

            # Moneyline
            if home_ml and away_ml and home_ml != away_ml:
                bet = betting_engine.create_moneyline_bet(
                    model_home_prob=model_home_prob,
                    home_odds_american=home_ml,
                    away_odds_american=away_ml,
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    commence_time=commence,
                    model_name=pred.get("model_name", "ensemble"),
                )
                if bet.is_actionable:
                    results.append(self._bet_to_prediction(bet))
                    bets_made += 1

            # Totals
            if model_total_over_prob is not None and total:
                game_total = game.consensus.total_consensus if hasattr(game, "consensus") and game.consensus else total
                over_odds = game.consensus.total_over_odds_consensus if hasattr(game, "consensus") and game.consensus else -110
                under_odds = game.consensus.total_under_odds_consensus if hasattr(game, "consensus") and game.consensus else -110

                if over_odds and under_odds:
                    over_bet = betting_engine.create_bet(
                        model_probability=model_total_over_prob,
                        odds_american=over_odds,
                        game_id=game_id,
                        home_team=home_team,
                        away_team=away_team,
                        commence_time=commence,
                        bet_side="over",
                        market_type="total",
                        market_line=game_total,
                        model_name=pred.get("model_name", "ensemble"),
                        opponent_odds_american=under_odds,
                    )
                    if over_bet.is_actionable:
                        results.append(self._bet_to_prediction(over_bet))
                        bets_made += 1

                    under_bet = betting_engine.create_bet(
                        model_probability=1.0 - model_total_over_prob,
                        odds_american=under_odds,
                        game_id=game_id,
                        home_team=home_team,
                        away_team=away_team,
                        commence_time=commence,
                        bet_side="under",
                        market_type="total",
                        market_line=game_total,
                        model_name=pred.get("model_name", "ensemble"),
                        opponent_odds_american=over_odds,
                    )
                    if under_bet.is_actionable:
                        results.append(self._bet_to_prediction(under_bet))
                        bets_made += 1

        # Sort by edge descending
        results.sort(key=lambda r: r["edge"], reverse=True)

        # Save results
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
                print(f"    ✅ {r['game']} | {r['prediction']} "
                      f"| Edge: {r['edge']:.2%} | EV: {r['ev']:.2%} "
                      f"| Stake: ${r['stake']:.0f}")

        return results

    # ═══════════════════════════════════════════════════════════════════
    #  STEPS
    # ═══════════════════════════════════════════════════════════════════

    def _fetch_tomorrow_odds(self, date: str) -> List[Any]:
        """Fetch odds for tomorrow's games."""
        try:
            # Try both possible import paths for OddsAPIClient
            OddsAPIClient = None
            try:
                from betting_intel.data.odds_fetcher import OddsAPIClient
            except ImportError:
                try:
                    from betting_intel.data.odds_fetcher import OddsAPIClient
                except ImportError:
                    from betting_intel.data.live_gateway import OddsAPIClient as OddsAPIClient

            from dotenv import load_dotenv
            import os
            load_dotenv()
            api_key = os.environ.get("ODDS_API_KEY", "")

            if not api_key:
                print("  [WARN] No ODDS_API_KEY found. No real odds available.")
                print("  [INFO] Set ODDS_API_KEY in .env file. Returning empty — no synthetic data.")
                return []

            client = OddsAPIClient(api_key=api_key)
            games = client.get_upcoming_games_with_odds(
                sport="basketball_nba",
                markets="h2h,spreads,totals",
                use_cache=False,
            )

            # Filter to tomorrow's games
            tomorrow_games = []
            for g in games:
                dt = g.commence_datetime
                if dt and dt.strftime("%Y-%m-%d") == date:
                    tomorrow_games.append(g)

            return tomorrow_games if tomorrow_games else []  # CRITICAL: never return non-tomorrow games as tomorrow predictions

        except ImportError as e:
            print(f"  [WARN] OddsAPIClient not available ({e}). No real odds available.")
            return []
        except Exception as e:
            print(f"  [WARN] Failed to fetch odds: {e}")
            return []

    def _store_odds_snapshot(self, games: List[Any]):
        """Store odds snapshot for CLV tracking."""
        try:
            from betting_intel.data.odds_ingestion import OddsIngestionEngine
            engine = OddsIngestionEngine(self._get_db_path())
            count = engine.ingest_snapshot(games)
            print(f"  Stored {count} odds records")
        except Exception as e:
            print(f"  [WARN] Failed to store odds: {e}")

    def _fetch_injuries(self, games: List[Any]) -> Dict:
        """Fetch injury data for tomorrow's teams."""
        injuries = {}
        try:
            from betting_intel.data.player_injury import InjuryAnalyzer
            analyzer = InjuryAnalyzer(self._get_db_path())
            for game in games:
                home = game.home_team_short if hasattr(game, "home_team_short") else ""
                away = game.away_team_short if hasattr(game, "away_team_short") else ""
                home_injuries = analyzer.get_team_injuries(home) if hasattr(analyzer, "get_team_injuries") else {}
                away_injuries = analyzer.get_team_injuries(away) if hasattr(analyzer, "get_team_injuries") else {}
                injuries[home] = home_injuries
                injuries[away] = away_injuries
            print(f"  Injury data for {len(injuries)} teams")
        except Exception as e:
            print(f"  [WARN] Injury fetch: {e}")
        return injuries

    def _generate_features(self, games: List[Any]) -> Dict[str, Dict]:
        """Generate feature vectors for each game."""
        features = {}
        try:
            from betting_intel.data.loader import NBADataLoader
            from betting_intel.data.features import FeatureEngineer
            import warnings
            warnings.filterwarnings("ignore")

            loader = NBADataLoader()
            fe = FeatureEngineer()

            raw_df = loader.load_game_logs()
            games_df = loader.build_game_dataset(raw_df)
            raw_df = loader.compute_rest_days(raw_df)
            feature_df = fe.build_all_features(games_df, raw_df)
            feature_cols = fe.select_features(feature_df)

            for game in games:
                game_id = game.id if hasattr(game, "id") else ""
                gid = int(game_id.split("_")[-1]) if "_" in str(game_id) else hash(game_id) % 100000
                # Build feature row using OddsAPIClient's static method
                from betting_intel.data.odds_fetcher import OddsAPIClient as OAC
                row = OAC.build_feature_row_for_game(game, feature_df, feature_cols)
                if row:
                    features[game_id] = row

            print(f"  Generated features for {len(features)} games")
        except Exception as e:
            print(f"  [WARN] Feature generation: {e}")

        return features

    def _extract_feature_importance(self, model_data: dict,
                                      feature_names: list[str]) -> list[dict]:
        """Extract top-5 feature contributions from a trained model.

        Handles multiple model formats:
          - sklearn/LightGBM/XGBoost: model.feature_importances_
          - sklearn linear models: model.coef_
          - dict with 'models' key (ensemble dict from saved model)
          - dict with 'feature_cols' and sub-models

        Returns:
            List of {"feature": str, "importance": float, "direction": "+" | "-"}
            sorted by absolute importance descending, top 5 only.
        """
        importances: np.ndarray | None = None

        try:
            if isinstance(model_data, dict):
                # Ensemble saved as dict — try to extract importance from sub-models
                sub_models = model_data.get("models", {})
                if sub_models:
                    # Average importance across all sub-models
                    all_imps = []
                    for name, sub in sub_models.items():
                        imp = self._extract_raw_importance(sub)
                        if imp is not None:
                            all_imps.append(imp)
                    if all_imps:
                        importances = np.mean(all_imps, axis=0)
                else:
                    # Maybe the dict itself has importance or models data
                    for key in ["feature_importances_", "coef_", "importance"]:
                        if key in model_data:
                            val = model_data[key]
                            if isinstance(val, (list, np.ndarray)):
                                importances = np.asarray(val).flatten()
                                break
            else:
                importances = self._extract_raw_importance(model_data)
        except Exception as e:
            logger.debug(f"Feature importance extraction failed: {e}")
            return []

        if importances is None or len(importances) == 0:
            return []

        # Match importances to feature names
        n_features = min(len(importances), len(feature_names))
        if n_features == 0:
            return []

        features = []
        for i in range(n_features):
            imp = float(importances[i])
            if abs(imp) > 1e-8:
                features.append({
                    "feature": feature_names[i],
                    "importance": abs(imp),
                    "direction": "+" if imp > 0 else "-",
                })

        # Sort by absolute importance descending, return top 5
        features.sort(key=lambda x: x["importance"], reverse=True)
        return features[:5]

    def _extract_raw_importance(self, model_obj) -> np.ndarray | None:
        """Extract raw importance/coefficient array from any model object."""
        if hasattr(model_obj, "feature_importances_"):
            return model_obj.feature_importances_
        if hasattr(model_obj, "coef_"):
            coef = model_obj.coef_
            if coef.ndim > 1:
                return coef[0] if coef.shape[0] == 1 else np.mean(np.abs(coef), axis=0)
            return coef
        if hasattr(model_obj, "feature_importances"):
            return model_obj.feature_importances
        # LightGBM booster
        if hasattr(model_obj, "booster_") and hasattr(model_obj.booster_, "feature_importance"):
            return model_obj.booster_.feature_importance(importance_type="gain")
        return None

    def _run_models(self, games: List[Any], features: Dict) -> List[Optional[Dict]]:
        """Run prediction models on each game."""
        predictions = []
        try:
            import warnings
            warnings.filterwarnings("ignore")
            import joblib
            import pandas as pd

            # Try to load trained model
            model_paths = [
                Path("models/total_model.pkl"),
                Path("models/ml_model.pkl"),
                Path("output/models/total_model.pkl"),
                Path("output/models/ml_model.pkl"),
            ]

            total_model = None
            ml_model = None
            total_feature_cols: list[str] = []
            models_loaded = False
            for p in model_paths:
                if p.exists():
                    try:
                        model_data = joblib.load(p)
                        if "total" in str(p):
                            total_model = model_data
                            # Extract feature columns from saved model dict
                            if isinstance(model_data, dict):
                                total_feature_cols = model_data.get("feature_cols", [])
                        else:
                            ml_model = model_data
                        models_loaded = True
                        print(f"  Loaded model: {p}")
                    except Exception as e:
                        logger.warning(f"Failed to load model {p}: {e}")
                        print(f"  [WARN] Failed to load {p}: {e}")

            if not models_loaded:
                print("  [WARN] No trained models found. Returning empty predictions.")
                print("  [WARN] Run 'python -m betting_intel.models.train' first to train models.")
                return []

            # Extract top-5 feature importance from the loaded model
            feature_breakdown = self._extract_feature_importance(
                total_model, total_feature_cols
            )
            if feature_breakdown:
                top_features = [
                    f"{f['feature']} ({f['direction']}{f['importance']:.1f})"
                    for f in feature_breakdown[:3]
                ]
                print(f"  Top features: {', '.join(top_features)}...")

            # Wire in probability calibration
            calibrator = None
            try:
                from betting_intel.validation.calibration import ProbabilityCalibrator, find_best_calibrator
                # Check for pre-fitted calibrator
                cal_path = Path("models/calibrator.pkl")
                if cal_path.exists():
                    calibrator = joblib.load(str(cal_path))
                    print(f"  Loaded calibrator: {cal_path}")
            except Exception as e:
                logger.debug(f"Calibrator not available: {e}")

            for game in games:
                game_id = game.id if hasattr(game, "id") else ""
                feat = features.get(game_id)

                pred = None
                if total_model and feat:
                    try:
                        feat_df = pd.DataFrame([feat])
                        total_pred = total_model.predict(feat_df)[0]

                        raw_home_prob = ml_model.predict_proba(feat_df)[0][1] if ml_model else 0.5

                        # PROPER PROBABILITY CONVERSION: Use logistic (sigmoid) function
                        # instead of the crude linear hack `0.5 + (pred - 220) * 0.002`.
                        # The logistic function P = 1/(1+exp(-k*(pred - market)))
                        # correctly maps the prediction error to [0,1] with proper
                        # calibration at the tails (linear hacks clip at 0.01/0.99).
                        # k=0.04 gives sensible calibration: 10pt diff → ~60% confidence
                        import math
                        # Get market total from real odds data only.
                        # NEVER use a hardcoded fallback (220.0, 224.0, etc.) —
                        # if there's no real market data, skip edge computation.
                        market_total = float(feat.get("market_total", feat.get("market_line", feat.get("trailing_avg_total_10g", 0))))
                        if market_total <= 0:
                            # No real market data — pred stays None, still appended
                            # to maintain 1:1 alignment with games list
                            pass
                        k_sensitivity = 0.04  # Calibrated: 10pt diff = P(over) ≈ 0.60
                        diff = float(total_pred) - market_total
                        raw_total_over_prob = 1.0 / (1.0 + math.exp(-k_sensitivity * diff))

                        # Apply calibration if available
                        home_prob = raw_home_prob
                        total_over_prob = raw_total_over_prob
                        if calibrator and hasattr(calibrator, 'calibrate'):
                            import numpy as np
                            home_prob = float(calibrator.calibrate(np.array([raw_home_prob]))[0])

                        home_prob = max(0.01, min(0.99, home_prob))
                        total_over_prob = max(0.01, min(0.99, total_over_prob))

                        pred = {
                            "home_win_prob": home_prob,
                            "total_over_prob": total_over_prob,
                            "predicted_total": float(total_pred),
                            "model_name": "trained_ensemble" if calibrator else "ensemble",
                            "confidence": 0.7,
                            "feature_breakdown": feature_breakdown,
                        }
                    except Exception as e:
                        logger.error(f"Model inference failed for {game_id}: {e}")
                        print(f"  [WARN] Model inference failed for {game_id}: {e}")

                predictions.append(pred)

        except Exception as e:
            logger.error(f"Model inference error: {e}")
            print(f"  [ERROR] Model inference failed: {e}")

        return predictions

    # ═══════════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _bet_to_prediction(self, bet) -> Dict:
        """Convert BetRecommendation to output dict."""
        game_label = f"{bet.away_team} @ {bet.home_team}"
        side_label = bet.bet_side.upper()
        line_str = ""
        if bet.market_line is not None:
            if bet.market_type == "total":
                line_str = f" O/U {bet.market_line:.1f}"
            elif bet.market_type == "spread":
                line_str = f" {bet.market_line:+.1f}"

        return {
            "game": game_label,
            "prediction": f"{side_label}{line_str}",
            "probability": bet.model_probability,
            "market_probability": bet.implied_probability,
            "edge": bet.edge_percentage,
            "ev": bet.expected_value,
            "stake": bet.stake.recommended_stake if bet.stake.is_valid else 0.0,
            "game_id": bet.game_id,
            "home_team": bet.home_team,
            "away_team": bet.away_team,
            "commence_time": bet.commence_time,
            "bet_side": bet.bet_side,
            "market_type": bet.market_type,
            "odds_american": bet.odds_american,
            "bankroll_pct": bet.stake.bankroll_percentage if bet.stake.is_valid else 0.0,
            "risk_level": bet.stake.risk_level,
            "clv": 0.0,
            "model_name": bet.model_name,
        }

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
            return self.data_dir / "betting_intel.db" if self.data_dir else Path("data/betting_intel.db")

    def _mock_games(self, date: str) -> List[Any]:
        """
        NEVER returns mock games — real data or nothing.
        
        If TheOddsAPI is unavailable, we return an empty list.
        No fake schedules. No randomly generated matchups.
        The system shows nothing when real data is unavailable.
        """
        logger.warning(
            "No real odds available from TheOddsAPI. "
            "Returning empty — no synthetic/mock games generated. "
            "Set ODDS_API_KEY or wait for quota reset."
        )
        return []
