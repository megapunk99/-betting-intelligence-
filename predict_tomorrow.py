"""
Generates betting predictions using the full advanced pipeline.
Integrates with TheOddsAPI for LIVE upcoming game predictions with
real market odds, edges, and staking recommendations.

Usage:
    # Live predictions for upcoming games (needs ODDS_API_KEY):
    python predict_tomorrow.py --live

    # Historical predictions (no API key needed):
    python predict_tomorrow.py

    # Demo live mode (uses demo odds if no API key):
    python predict_tomorrow.py --live --demo
"""

import sys
import os
import warnings
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DB_PATH, INITIAL_BANKROLL, MIN_EDGE_THRESHOLD, UNIT_SIZE,
    ODDS_API_KEY, ODDS_CACHE_TTL_MINUTES, ODDS_DEFAULT_MARKETS,
    ODDS_DEFAULT_REGIONS, FAST_MODE,
)
from data.loader import NBADataLoader
from data.features import FeatureEngineer
from data.odds_fetcher import (
    OddsAPIClient, OddsGame, display_odds_card,
    SHORT_NAME_TO_TEAM_ID, ODDS_TO_SHORT_NAME,
)
from models.predictors import (
    TotalPointsPredictor, SpreadPredictor, MomentumModel,
    StackingEnsemblePredictor, create_best_model, create_tuned_lgbm_regressor
)
from backtesting.engine import WalkForwardEngine, BacktestResult
from betting.edge import EdgeDetector
from betting.bankroll import BankrollManager
from src.betting_intel.betting.monte_carlo import MonteCarloSimulator
from backtesting.metrics import BacktestMetrics


class AdvancedPredictionEngine:
    """
    End-to-end prediction engine that:
    1. Loads latest data
    2. Engineers advanced v2.0 features
    3. Trains ensemble of state-of-the-art models
    4. Generates calibrated predictions with uncertainty
    5. Simulates outcomes via Monte Carlo
    6. Produces actionable betting recommendations

    When live_mode=True, fetches actual upcoming games from TheOddsAPI
    and compares model predictions to real market lines.
    """

    def __init__(self, tune_hyperparams: bool = True, live_mode: bool = False,
                 demo_mode: bool = False):
        self.loader = NBADataLoader()
        self.feature_engineer = FeatureEngineer()
        self.tune = tune_hyperparams and os.environ.get("SKIP_TUNE", "").lower() != "true"
        self.live_mode = live_mode
        # Demo mode if live_mode and no valid API key
        self.demo_mode = demo_mode
        if live_mode and not demo_mode:
            key_valid = ODDS_API_KEY and ODDS_API_KEY != "your-api-key-here" and len(ODDS_API_KEY) > 10
            if not key_valid:
                print("  [Info] No valid ODDS_API_KEY found. Running in demo mode.")
                print("     Set ODDS_API_KEY in config.py or as env variable for real data.")
                print("     Get a free key at: https://the-odds-api.com/\n")
                self.demo_mode = True
        self.odds_client = OddsAPIClient(
            api_key=ODDS_API_KEY,
            cache_ttl_minutes=ODDS_CACHE_TTL_MINUTES,
        ) if live_mode else None
        self.results = {}

    def run(self) -> Dict:
        """Run the full prediction pipeline.

        In live_mode, fetches actual upcoming games from TheOddsAPI
        and generates predictions with real market comparisons.
        """
        header = "BETTING INTELLIGENCE v2.0 — LIVE PREDICTION ENGINE" if self.live_mode else \
                 "BETTING INTELLIGENCE v2.0 — ADVANCED PREDICTION ENGINE"
        tagline = "Real market odds from TheOddsAPI" if self.live_mode else \
                  "Professional-grade NBA betting analytics"

        header_line = "=" * 70
        print(header_line)
        print(f"  {header}")
        print(f"  {tagline}")
        print(header_line)

        # ── 1. Load & Prepare Data ──────────────────────────────────────
        print("\n[1/6] Loading NBA data...")
        raw_df = self.loader.load_game_logs()
        games_df = self.loader.build_game_dataset(raw_df)
        raw_df = self.loader.compute_rest_days(raw_df)
        print(f"  Games: {len(games_df)} | Date: {games_df['GAME_DATE'].min().date()} to {games_df['GAME_DATE'].max().date()}")

        # ── 2. Engineer v2.0 Advanced Features ──────────────────────────
        print("\n[2/6] Engineering advanced features (v2.0)...")
        print("  Including: Elo ratings, TS%, opponent-adjusted, travel fatigue")
        feature_df = self.feature_engineer.build_all_features(games_df, raw_df)
        feature_cols = self.feature_engineer.select_features(feature_df)
        print(f"  Features created: {len(feature_cols)}")

        # Clean data
        clean_df = feature_df.dropna(subset=feature_cols, thresh=len(feature_cols) // 2).copy()
        clean_df = clean_df.reset_index(drop=True)
        print(f"  Clean samples: {len(clean_df)}")

        # ── 3. Train Advanced Models ────────────────────────────────────
        print("\n[3/6] Training advanced models (v2.0)...")
        # Ensure no NaN values in feature matrix
        clean_df[feature_cols] = clean_df[feature_cols].fillna(0).infer_objects(copy=False)
        clean_df = clean_df.replace([np.inf, -np.inf], 0)
        models = self._train_models(clean_df, feature_cols)
        self.results["models"] = models

        # ── 4. Generate Predictions ─────────────────────────────────────
        if self.live_mode:
            print("\n[4/6] Fetching live odds & generating predictions...")
            predictions, odds_games = self._generate_live_predictions(
                clean_df, feature_cols, models
            )
            self.results["odds_games"] = odds_games
        else:
            print("\n[4/6] Generating predictions for recent games...")
            predictions = self._generate_predictions(clean_df, feature_cols, models)
        self.results["predictions"] = predictions

        # ── 5. Monte Carlo Simulation ───────────────────────────────────
        print("\n[5/6] Running Monte Carlo simulation...")
        mc_results = self._run_monte_carlo(models, clean_df)
        self.results["monte_carlo"] = mc_results

        # ── 6. Generate Actionable Recommendations ─────────────────────
        print("\n[6/6] Generating betting recommendations...")
        recommendations = self._generate_recommendations(predictions, mc_results)
        self.results["recommendations"] = recommendations

        # Display results
        self._display_results(predictions, recommendations, mc_results)

        # Save to file
        self._save_results()

        print("\n" + "=" * 70)
        print("  PREDICTION ENGINE COMPLETE")
        print("=" * 70)

        return self.results

    def _train_models(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Train models using walk-forward validation.

        When FAST_MODE is True, only trains LightGBM + Momentum
        for quick results. Otherwise trains the full ensemble.
        """
        model_results = {}

        # ── Total Points Prediction (REGRESSION) ──
        if FAST_MODE:
            print("  Training total points model (FAST_MODE: LightGBM only)...")
        else:
            print("  Training Total Points models (full ensemble)...")

        X = df[feature_cols].values
        y_total = df["total_points"].values
        split = int(len(X) * 0.7)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y_total[:split], y_total[split:]

        # LightGBM (tuned)
        if self.tune:
            print("    Tuning LightGBM with Optuna...")
            lgbm_model = create_tuned_lgbm_regressor(X_train, y_train, n_trials=30)
        else:
            lgbm_model = TotalPointsPredictor("lightgbm")
        lgbm_model.fit(X_train, y_train)
        lgbm_metrics = lgbm_model.evaluate(X_test, y_test)
        model_results["total_lgbm"] = {"model": lgbm_model, "metrics": lgbm_metrics}
        print(f"    LightGBM: MAE={lgbm_metrics['mae']:.1f}, R2={lgbm_metrics['r2']:.3f}")

        if not FAST_MODE:
            # CatBoost
            cb_model = TotalPointsPredictor("catboost")
            cb_model.fit(X_train, y_train)
            cb_metrics = cb_model.evaluate(X_test, y_test)
            model_results["total_catboost"] = {"model": cb_model, "metrics": cb_metrics}
            print(f"    CatBoost: MAE={cb_metrics['mae']:.1f}, R2={cb_metrics['r2']:.3f}")

            # Bayesian Ridge (uncertainty)
            br_model = TotalPointsPredictor("bayesian")
            br_model.fit(X_train, y_train)
            br_metrics = br_model.evaluate(X_test, y_test)
            model_results["total_bayesian"] = {"model": br_model, "metrics": br_metrics}
            print(f"    BayesianRidge: MAE={br_metrics['mae']:.1f}, R2={br_metrics['r2']:.3f}")

            # Stacking Ensemble
            ensemble = StackingEnsemblePredictor("regression")
            ensemble.add_base_model(lgbm_model)
            ensemble.add_base_model(cb_model)
            ensemble.add_base_model(br_model)
            try:
                ensemble.fit(X_train, y_train)
                ensemble_preds = ensemble.predict(X_test)
                ensemble_mae = np.mean(np.abs(ensemble_preds - y_test))
                ensemble_r2 = 1 - np.sum((ensemble_preds - y_test)**2) / np.sum((y_test - y_test.mean())**2)
                model_results["total_ensemble"] = {"model": ensemble, "metrics": {"mae": ensemble_mae, "r2": ensemble_r2}}
                print(f"    StackingEnsemble: MAE={ensemble_mae:.1f}, R2={ensemble_r2:.3f}")
            except Exception as e:
                print(f"    [!] Ensemble failed: {e}")

        # ── Momentum/Classification Model ──
        print("  Training Momentum (classification) model...")
        df["home_win"] = (df["point_diff"] > 0).astype(int)
        y_class = df["home_win"].values[:len(X)]  # Use same split

        momentum_features = [c for c in feature_cols if any(
            kw in c for kw in ["streak", "momentum", "win_pct", "margin_volatility",
                               "rest_", "elo_", "form_", "weighted_", "win_prob",
                               "fatigue", "travel", "tz_", "avg_pm_", "net_rating",
                               "mom_vs_opp", "sos_", "home_advantage"]
        )]
        if len(momentum_features) < 5:
            momentum_features = feature_cols

        X_mom = df[momentum_features].values[:len(X)]
        split_m = int(len(X_mom) * 0.7)
        Xm_train, Xm_test = X_mom[:split_m], X_mom[split_m:]
        ym_train, ym_test = y_class[:split_m], y_class[split_m:]

        momentum_model = MomentumModel("lightgbm", calibrate=True)
        momentum_model.fit(Xm_train, ym_train)
        mom_metrics = momentum_model.evaluate(Xm_test, ym_test)
        model_results["momentum"] = {"model": momentum_model, "features": momentum_features, "metrics": mom_metrics}
        print(f"    Momentum: Acc={mom_metrics['accuracy']:.1%}, ROC-AUC={mom_metrics.get('roc_auc', 0):.3f}")

        return model_results

    def _generate_predictions(self, df: pd.DataFrame, feature_cols: List[str],
                               models: Dict) -> List[Dict]:
        """Generate predictions for the most recent/upcoming games."""
        predictions = []

        # Get the most recent games for prediction context
        recent_df = df.sort_values("GAME_DATE").tail(50).copy()

        # Use the best total points model (lowest MAE)
        total_models = {k: v for k, v in models.items() if k.startswith("total_")}
        best_model_key = min(total_models, key=lambda k: total_models[k]["metrics"]["mae"])
        best_model = total_models[best_model_key]["model"]
        print(f"  Best total model: {best_model_key} (MAE={total_models[best_model_key]['metrics']['mae']:.1f})")

        # Momentum model
        momentum_model = models.get("momentum", {}).get("model")
        momentum_features = models.get("momentum", {}).get("features", feature_cols)

        # Generate predictions for recent games to show model performance
        for idx in range(len(recent_df)):
            row = recent_df.iloc[idx]
            game_features = row[feature_cols].values.reshape(1, -1)

            try:
                total_pred = best_model.predict(game_features)[0]

                # Get uncertainty estimate
                if hasattr(best_model, "predict_with_uncertainty"):
                    _, uncertainty = best_model.predict_with_uncertainty(game_features)
                    pred_std = uncertainty[0]
                else:
                    pred_std = total_models[best_model_key]["metrics"]["mae"]

                # Market line baseline: use trailing average as proxy for sportsbook's line
                # This is deliberately NOT a feature the model sees during training.
                market_line = row.get(
                    "market_line_baseline",
                    row.get("trailing_avg_total_10g",
                            row.get("avg_pts_5g_home", 110) + row.get("avg_pts_5g_away", 105))
                )
                market_line = float(market_line) if not np.isnan(market_line) else 220.0

                edge_pct = (total_pred - market_line) / max(market_line, 1) if market_line > 0 else 0.0

                # Momentum probability
                home_win_prob = 0.5
                if momentum_model is not None:
                    mom_features = row[momentum_features].values.reshape(1, -1)
                    try:
                        proba = momentum_model.predict_proba(mom_features)
                        home_win_prob = float(proba[0, 1])
                    except Exception:
                        pass

                predictions.append({
                    "game_date": str(row["GAME_DATE"].date()),
                    "game_id": str(row["GAME_ID"]),
                    "matchup": f"{row.get('TEAM_NAME_home', '?')} vs {row.get('TEAM_NAME_away', '?')}",
                    "home_team": row.get("TEAM_NAME_home", "?"),
                    "away_team": row.get("TEAM_NAME_away", "?"),
                    "predicted_total": round(float(total_pred), 1),
                    "market_line": round(float(market_line), 1),
                    "edge_pct": round(float(edge_pct), 4),
                    "prediction_std": round(float(pred_std), 1),
                    "home_win_prob": round(float(home_win_prob), 4),
                    "elo_home": float(row.get("elo_home_pre", 1500)),
                    "elo_away": float(row.get("elo_away_pre", 1500)),
                    "rest_home": int(row.get("rest_home_days", 3)),
                    "rest_away": int(row.get("rest_away_days", 3)),
                    "travel_distance": int(row.get("travel_distance", 0)),
                    "home_win_pct_10": float(row.get("win_pct_10g_home", 0.5)),
                    "away_win_pct_10": float(row.get("win_pct_10g_away", 0.5)),
                })
            except Exception as e:
                print(f"    [!] Prediction failed for game {row.get('GAME_ID', '?')}: {e}")

        return predictions

    def _generate_live_predictions(
        self, df: pd.DataFrame, feature_cols: List[str], models: Dict
    ) -> Tuple[List[Dict], List[OddsGame]]:
        """
        Fetch upcoming games from TheOddsAPI and generate predictions
        with real market lines for edge detection.
        """
        predictions = []

        # ── 1. Fetch upcoming games from TheOddsAPI ─────────────────────
        print("\n  [Fetching live odds from TheOddsAPI...]")
        odds_games = self.odds_client.get_upcoming_games_with_odds(
            use_cache=not self.demo_mode  # Don't cache demo games
        )
        print(f"  Found {len(odds_games)} upcoming NBA games")


        if not odds_games:
            print("  [!] No upcoming games found. Check API key or try --demo mode.")
            return predictions, odds_games

        # Quick summary of fetched games
        display_odds_card(odds_games, title="UPCOMING NBA GAMES (TheOddsAPI)")

        # ── 2. Get the best models ──────────────────────────────────────
        total_models = {k: v for k, v in models.items() if k.startswith("total_")}
        if not total_models:
            print("  [!] No trained models available")
            return predictions, odds_games

        best_model_key = min(total_models, key=lambda k: total_models[k]["metrics"]["mae"])
        best_model = total_models[best_model_key]["model"]
        best_model_mae = total_models[best_model_key]["metrics"]["mae"]

        momentum_model = models.get("momentum", {}).get("model")
        momentum_features = models.get("momentum", {}).get("features", feature_cols)

        print(f"\n  Best total model: {best_model_key} (MAE={best_model_mae:.1f})")

        # ── 3. Build feature vectors & predict for each game ────────────
        print("\n  Generating predictions for each upcoming game...")
        for i, game in enumerate(odds_games, 1):
            try:
                # Build feature row for this upcoming matchup
                feature_row = OddsAPIClient.build_feature_row_for_game(
                    game, df, feature_cols
                )

                if feature_row is None:
                    print(f"  [{i}/{len(odds_games)}] {game.matchup:45s}  [Skip] Insufficient historical data")
                    continue

                # Create feature array for prediction
                X_pred = np.array([[feature_row.get(c, 0) for c in feature_cols]])

                # Total points prediction
                total_pred = float(best_model.predict(X_pred)[0])

                # Uncertainty estimate
                if hasattr(best_model, "predict_with_uncertainty"):
                    _, uncertainty = best_model.predict_with_uncertainty(X_pred)
                    pred_std = float(uncertainty[0])
                else:
                    pred_std = best_model_mae

                # Market total from API
                market_line = game.market_total or (game.total_over if game.total_over else game.total_under or 220.0)
                market_total = float(market_line)

                # Edge over market
                edge_pct = (total_pred - market_total) / max(market_total, 1)

                # Home win probability from momentum model
                home_win_prob = game.implied_home_win_prob or 0.5
                if momentum_model is not None:
                    try:
                        X_mom = np.array([[feature_row.get(c, 0) for c in momentum_features]])
                        proba = momentum_model.predict_proba(X_mom)
                        model_home_prob = float(proba[0, 1])
                        # Blend with market implied probability
                        home_win_prob = 0.6 * model_home_prob + 0.4 * (game.implied_home_win_prob or 0.5)
                    except Exception:
                        pass

                # Game time
                game_date_str = game.commence_datetime.strftime("%Y-%m-%d %H:%M") if game.commence_datetime else game.commence_time

                predictions.append({
                    "game_date": game_date_str,
                    "game_id": game.id,
                    "matchup": game.matchup,
                    "home_team": game.home_team_short,
                    "away_team": game.away_team_short,
                    "predicted_total": round(total_pred, 1),
                    "market_line": round(market_total, 1),
                    "edge_pct": round(edge_pct, 4),
                    "prediction_std": round(pred_std, 1),
                    "home_win_prob": round(home_win_prob, 4),
                    "implied_home_win": game.implied_home_win_prob or 0.5,
                    "market_moneyline": f"{game.home_moneyline:+d}" if game.home_moneyline else "N/A",
                    "market_spread": game.home_spread,
                    "elo_home": float(feature_row.get("elo_home_pre", 1500)),
                    "elo_away": float(feature_row.get("elo_away_pre", 1500)),
                    "rest_home": int(feature_row.get("rest_home_days", 3)),
                    "rest_away": int(feature_row.get("rest_away_days", 3)),
                    "travel_distance": int(feature_row.get("travel_distance", 0)),
                    "home_win_pct_10": float(feature_row.get("win_pct_10g_home", 0.5)),
                    "away_win_pct_10": float(feature_row.get("win_pct_10g_away", 0.5)),
                    "is_live": True,
                    "source": "TheOddsAPI",
                })

                edge_str = f"{edge_pct:+.1%} edge" if abs(edge_pct) > 0.01 else "no edge"
                print(f"  [{i}/{len(odds_games)}] {game.matchup:45s}  > {total_pred:.0f} vs {market_total:.0f} ({edge_str})")

            except Exception as e:
                print(f"  [{i}/{len(odds_games)}] {game.matchup:45s}  [Error] {e}")
                continue

        print(f"\n  Generated predictions for {len(predictions)}/{len(odds_games)} games")
        return predictions, odds_games

    def _run_monte_carlo(self, models: Dict, df: pd.DataFrame) -> Optional[Dict]:
        """Run Monte Carlo simulation to estimate strategy variance."""
        # Check if we have backtest results to simulate from
        if "momentum" not in models:
            return None

        # Simulate win rate uncertainty based on momentum model
        mom_metrics = models["momentum"]["metrics"]
        win_rate = mom_metrics.get("accuracy", 0.55)
        n_bets = 500  # Simulate a full season

        simulator = MonteCarloSimulator(n_simulations=5000)
        sim_result = simulator.simulate_win_rate_only(n_bets=n_bets, true_win_rate=win_rate)

        print(f"  Win rate uncertainty ({win_rate:.1%} assumed, {n_bets} bets):")
        print(f"    95% CI: {sim_result['ci_95'][0]:.1%} to {sim_result['ci_95'][1]:.1%}")
        print(f"    P(profitable): {sim_result['prob_profitable']:.1%}")

        return sim_result

    def _generate_recommendations(self, predictions: List[Dict],
                                   mc_results: Optional[Dict]) -> List[Dict]:
        """Generate actionable betting recommendations from predictions."""
        recommendations = []

        for p in predictions:
            edge = abs(p["edge_pct"])

            if edge < MIN_EDGE_THRESHOLD:
                continue

            # Determine bet type based on edge sign
            if p["edge_pct"] > 0:
                bet_side = "OVER"
                direction = f"OVER {p['market_line']}"
            else:
                bet_side = "UNDER"
                direction = f"UNDER {p['market_line']}"

            # Confidence based on edge size and prediction std
            edge_quality = edge / max(p.get("prediction_std", 5) / 10, 0.01)
            elo_confidence = abs(p.get("elo_home", 1500) - p.get("elo_away", 1500)) / 200
            form_confidence = abs(p.get("home_win_pct_10", 0.5) - p.get("away_win_pct_10", 0.5))

            confidence_score = min(1.0, edge_quality * 0.4 + elo_confidence * 0.3 + form_confidence * 0.3)

            if confidence_score >= 0.65:
                confidence_label = "HIGH"
            elif confidence_score >= 0.5:
                confidence_label = "MEDIUM"
            else:
                confidence_label = "LOW"

            # Kelly stake calculation
            kelly_fraction = min(edge * 10, 0.25) * confidence_score
            stake_dollars = kelly_fraction * INITIAL_BANKROLL

            # Only include bets with decent confidence
            if confidence_score < 0.35:
                continue

            rec = {
                "game_date": p["game_date"],
                "game_id": p["game_id"],
                "matchup": p["matchup"],
                "bet_type": f"TOTAL {direction}",
                "predicted_total": p["predicted_total"],
                "market_line": p["market_line"],
                "edge_pct": round(edge, 3),
                "confidence": confidence_label,
                "confidence_score": round(confidence_score, 2),
                "suggested_stake": f"${stake_dollars:,.0f}",
                "kelly_pct": f"{kelly_fraction * 100:.1f}%",
                "home_win_prob": p.get("home_win_prob", 0.5),
                "analysis_factors": {
                    "rest_days_home": p.get("rest_home"),
                    "rest_days_away": p.get("rest_away"),
                    "travel_miles": p.get("travel_distance"),
                    "elo_home": int(p.get("elo_home", 1500)),
                    "elo_away": int(p.get("elo_away", 1500)),
                    "home_form_10g": p.get("home_win_pct_10"),
                    "away_form_10g": p.get("away_win_pct_10"),
                },
                "reasoning": self._generate_reasoning(p, edge, confidence_label),
            }

            # Copy live-specific fields from prediction to recommendation
            if p.get("is_live"):
                rec["is_live"] = True
                rec["market_moneyline"] = p.get("market_moneyline", "")
                rec["market_spread"] = p.get("market_spread")
                rec["source"] = p.get("source", "TheOddsAPI")

            recommendations.append(rec)

        # Sort by edge (descending)
        recommendations.sort(key=lambda r: r["edge_pct"], reverse=True)

        return recommendations

    def _generate_reasoning(self, prediction: Dict, edge: float,
                             confidence: str) -> str:
        """Generate human-readable reasoning for a prediction."""
        factors = []

        # Elo difference
        elo_diff = prediction.get("elo_home", 1500) - prediction.get("elo_away", 1500)
        if abs(elo_diff) > 50:
            stronger = "Home" if elo_diff > 0 else "Away"
            factors.append(f"{stronger} team has significant Elo advantage ({abs(elo_diff):.0f} pts)")

        # Rest advantage
        rest_diff = prediction.get("rest_home", 3) - prediction.get("rest_away", 3)
        if abs(rest_diff) >= 1:
            rested = "Home" if rest_diff > 0 else "Away"
            factors.append(f"{rested} team has {'+{:.0f}'.format(abs(rest_diff))} rest advantage")

        # Travel
        travel = prediction.get("travel_distance", 0)
        if travel > 1500:
            factors.append(f"Away team traveling {travel} miles (significant fatigue factor)")

        # Form
        home_form = prediction.get("home_win_pct_10", 0.5)
        away_form = prediction.get("away_win_pct_10", 0.5)
        form_diff = abs(home_form - away_form)
        if form_diff > 0.15:
            better = "Home" if home_form > away_form else "Away"
            factors.append(f"{better} team in better form ({max(home_form, away_form):.0%} last 10)")

        # Edge size
        if edge > 0.04:
            factors.append(f"Strong edge detected ({edge:.1%})")
        elif edge > 0.03:
            factors.append(f"Moderate edge detected ({edge:.1%})")

        # Market prediction
        if prediction["predicted_total"] > prediction["market_line"]:
            factors.append(f"Model predicts {prediction['predicted_total']:.0f} pts vs market {prediction['market_line']:.0f}")

        if not factors:
            factors.append("Statistical model identifies marginal edge")

        return "; ".join(factors) if factors else "No significant factors identified"

    def _display_results(self, predictions: List[Dict],
                          recommendations: List[Dict],
                          mc_results: Optional[Dict]):
        """Display formatted prediction results."""
        is_live = self.live_mode

        print("\n" + "=" * 95)
        if is_live:
            print("  ** ACTIONABLE BETS FOR UPCOMING GAMES -- RANKED BY EDGE **")
        else:
            print("  ** RECOMMENDED BETS -- RANKED BY EDGE **")
        print("=" * 95)

        if not recommendations:
            print("\n  No actionable bets found with current edge thresholds.")
            print("  Try lowering MIN_EDGE_THRESHOLD in config.py")
        else:
            print(f"\n{'#':<3} {'Game':<38} {'Pred':<8} {'Market':<8} {'Edge':<8} {'Kelly':<8} {'Conf':<8} {'ML/Sprd':<12}")
            print("-" * 95)
            for i, rec in enumerate(recommendations, 1):
                bet_type = rec['bet_type'][:36]
                ml_spread = ""
                if is_live and "market_moneyline" in rec:
                    ml_spread = rec.get("market_moneyline", "")
                    if rec.get("market_spread"):
                        spread = rec["market_spread"]
                        sp_str = f"{spread:+d}" if spread == int(spread) else f"{spread:+.0f}"
                        ml_spread = f"{sp_str}" if ml_spread == "" else f"{sp_str}"
                print(f"{i:<3} {bet_type:<38} "
                      f"{rec['predicted_total']:<8} "
                      f"{rec['market_line']:<8} "
                      f"{rec['edge_pct']:.1%}   "
                      f"{rec['kelly_pct']:<8} "
                      f"{rec['confidence']:<8} "
                      f"{ml_spread:<12}")

            total_stake = sum(float(r['suggested_stake'].replace('$', '').replace(',', ''))
                             for r in recommendations)
            print("-" * 95)
            print(f"  TOTAL STAKING: ${total_stake:,.0f} ({total_stake/INITIAL_BANKROLL:.1%} of bankroll)")
            print(f"  HIGH confidence: {sum(1 for r in recommendations if r['confidence'] == 'HIGH')} bets")
            print(f"  MEDIUM confidence: {sum(1 for r in recommendations if r['confidence'] == 'MEDIUM')} bets")

        # Show all games with market odds
        # Build a lookup from game_id -> kelly_pct for the detailed display
        rec_kelly_lookup: Dict[str, str] = {}
        for r in recommendations:
            gid = r.get("game_id", "")
            if gid:
                rec_kelly_lookup[gid] = r["kelly_pct"]

        print("\n" + "=" * 95)
        header = "DETAILED ANALYSIS -- ALL UPCOMING GAMES WITH ODDS" if is_live else \
                 "DETAILED ANALYSIS -- TOP PREDICTIONS"
        print(f"  {header}")
        print("=" * 95)
        for i, p in enumerate(predictions[:8], 1):
            game_id = p.get("game_id", "")
            kelly_pct = rec_kelly_lookup.get(game_id, "0.0%")

            if p.get("is_live"):
                # Live mode display
                moneyline = p.get("market_moneyline", "")
                spread = p.get("market_spread", "")
                print(f"\n  [{i}] {p['matchup']}")
                print(f"      {'Game Time:':16s} {p['game_date']}")
                print(f"      {'Model Total:':16s} {p['predicted_total']:.1f} | "
                      f"Market Total: {p['market_line']:.1f} | "
                      f"Edge: {p['edge_pct']:.1%}")
                print(f"      {'ML:':16s} {moneyline:<10} | "
                      f"Spread: {str(spread):<6} | "
                      f"Home Win: {p['home_win_prob']:.0%} | "
                      f"Kelly: {kelly_pct}")
                print(f"      {'Elo:':16s} {p.get('elo_home', 0):.0f} vs {p.get('elo_away', 0):.0f} | "
                      f"Rest: {p.get('rest_home', 0)}d vs {p.get('rest_away', 0)}d | "
                      f"Travel: {p.get('travel_distance', 0):,}mi")
                print(f"      {'Source:':16s} {p.get('source', 'NBA DB')} | "
                      f"Uncertainty: ±{p.get('prediction_std', 0):.0f} pts")
            else:
                # Historical mode display
                print(f"\n  [{i}] {p['matchup']} ({p['game_date']})")
                print(f"      Predicted Total: {p['predicted_total']:.1f} | Market: {p['market_line']:.1f}")
                print(f"      Edge: {p['edge_pct']:.1%} | Home Win Prob: {p.get('home_win_prob', 0.5):.0%}")
                print(f"      Elo: {p.get('elo_home', 1500):.0f} vs {p.get('elo_away', 1500):.0f} | "
                      f"Travel: {p.get('travel_distance', 0):,}mi | "
                      f"Rest: {p.get('rest_home', 0)}d vs {p.get('rest_away', 0)}d")

        if mc_results:
            print(f"\n  Monte Carlo: 95% CI win rate = {mc_results['ci_95'][0]:.1%} to {mc_results['ci_95'][1]:.1%} | "
                  f"P(profitable) = {mc_results['prob_profitable']:.1%}")

        # Final summary
        disclaimer = "These predictions use v2.0 advanced models with Elo, TS%,"
        if is_live:
            disclaimer += "\n  opponent-adjusted stats, travel fatigue, and LIVE market odds from TheOddsAPI."
        else:
            disclaimer += "\n  opponent-adjusted stats, travel fatigue, and Monte Carlo simulation."
        print("\n" + "=" * 95)
        print("  RISK DISCLAIMER")
        print("=" * 95)
        print(f"  {disclaimer}")
        print("  Past performance does not guarantee future results.")
        print(f"  Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
        print("=" * 95)

    def _save_results(self):
        """Save all results to disk."""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save predictions
        if self.results.get("predictions"):
            pred_path = output_dir / f"predictions_v2_{timestamp}.json"
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump({
                    "generated_at": datetime.now().isoformat(),
                    "engine_version": "2.0",
                    "mode": "live" if self.live_mode else "historical",
                    "models_used": [
                        k for k in self.results.get("models", {}).keys()
                        if k != "momentum"
                    ],
                    "momentum_model": {
                        "accuracy": self.results.get("models", {}).get("momentum", {}).get("metrics", {}).get("accuracy", 0),
                        "roc_auc": self.results.get("models", {}).get("momentum", {}).get("metrics", {}).get("roc_auc", 0),
                    },
                    "predictions": self.results["predictions"],
                }, f, indent=2, default=str)
            print(f"\n  Predictions saved: {pred_path.name}")

        # Save recommendations
        if self.results.get("recommendations"):
            rec_path = output_dir / f"recommendations_v2_{timestamp}.json"
            with open(rec_path, "w", encoding="utf-8") as f:
                json.dump({
                    "generated_at": datetime.now().isoformat(),
                    "engine_version": "2.0",
                    "mode": "live" if self.live_mode else "historical",
                    "bankroll": INITIAL_BANKROLL,
                    "recommendations": self.results["recommendations"],
                }, f, indent=2, default=str)
            print(f"  Recommendations saved: {rec_path.name}")

        # Save odds games (live mode only)
        if self.live_mode and self.results.get("odds_games"):
            odds_path = output_dir / f"odds_upcoming_v2_{timestamp}.json"
            with open(odds_path, "w", encoding="utf-8") as f:
                json.dump({
                    "generated_at": datetime.now().isoformat(),
                    "source": "TheOddsAPI",
                    "games": [g.to_dict() for g in self.results["odds_games"]],
                }, f, indent=2, default=str)
            print(f"  Odds games saved: {odds_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Betting Intelligence v2.0 — NBA Prediction Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_tomorrow.py                          # Historical mode (existing games)
  python predict_tomorrow.py --live                   # Live predictions for upcoming games
  python predict_tomorrow.py --live --demo             # Demo mode (no API key needed)
  python predict_tomorrow.py --live --no-tune          # Skip hyperparameter tuning (faster)
  python predict_tomorrow.py --no-tune                # Skip tuning in historical mode
        """
    )
    parser.add_argument("--live", action="store_true",
                        help="Fetch real upcoming games from TheOddsAPI and predict")
    parser.add_argument("--demo", action="store_true",
                        help="Use demo data (no API key needed, only with --live)")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip Optuna hyperparameter tuning (faster startup)")
    parser.add_argument("--min-edge", type=float, default=None,
                        help=f"Override minimum edge threshold (default: {MIN_EDGE_THRESHOLD})")

    args = parser.parse_args()

    mode = "LIVE" if args.live else "HISTORICAL"
    mode_tag = "(demo)" if args.demo else ""

    print("=" * 60)
    print(f"  BETTING INTELLIGENCE SYSTEM v2.0")
    print(f"  Mode: {mode} {mode_tag}".strip())
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    engine = AdvancedPredictionEngine(
        tune_hyperparams=not args.no_tune,
        live_mode=args.live,
        demo_mode=args.demo,
    )
    results = engine.run()

    return results


if __name__ == "__main__":
    main()
