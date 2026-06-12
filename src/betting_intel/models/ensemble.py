"""
Ensemble Model System — Phase 2.7 of the Professional Betting Intelligence Platform.

Do not rely on a single model. Multiple models capture different signal types.

Models:
    1. ELO Model — rating-system based, captures team strength evolution
    2. XGBoost Model — gradient boosted trees, captures non-linear interactions
    3. LightGBM Model — faster gradient boosting, handles categorical features
    4. Logistic Regression — interpretable baseline, captures linear patterns
    5. Market-Based Model — extracts signal from market consensus itself

Ensemble:
    - Weighted average of model probabilities
    - Weights learned from recent out-of-sample performance
    - Adaptive: weights shift as models drift

Output:
    final_probability — the ensemble's best estimate
    model_weights — transparent breakdown of contributions
    consensus_level — how much the models agree (0-1)
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelPrediction:
    """Output from a single model."""
    name: str
    probability: float
    confidence: float        # 0-1, model's self-assessment
    weight: float = 1.0      # Current ensemble weight


@dataclass
class EnsemblePrediction:
    """Final ensemble prediction with full breakdown."""
    game_id: str = ""
    home_team: str = ""
    away_team: str = ""

    # Total prediction
    home_win_probability: float = 0.5
    total_over_probability: Optional[float] = None
    predicted_total: Optional[float] = None
    home_cover_probability: Optional[float] = None

    # Model breakdown
    model_predictions: List[ModelPrediction] = field(default_factory=list)
    model_weights: Dict[str, float] = field(default_factory=dict)

    # Consensus
    consensus_level: float = 0.0    # 0 = total disagreement, 1 = unanimous
    num_models: int = 0
    model_variance: float = 0.0

    # Metadata
    generated_at: str = ""
    model_name: str = "ensemble"


# ═══════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL MODELS
# ═══════════════════════════════════════════════════════════════════════════

class BaseModel:
    """Base class for all prediction models."""
    name: str = "base"

    def predict(self, features: Dict) -> ModelPrediction:
        raise NotImplementedError

    def predict_batch(self, feature_rows: List[Dict]) -> List[ModelPrediction]:
        return [self.predict(f) for f in feature_rows]

    def get_params(self) -> Dict:
        return {}


class ELOModel(BaseModel):
    """
    ELO-based rating system for NBA team strength.

    Each team has an ELO rating. The expected win probability is:
        P(home_win) = 1 / (1 + 10^((away_elo - home_elo + home_advantage) / 400))

    Home court advantage: ~100 ELO points (~60% win prob for equal teams)
    K-factor (rating update speed): 20 (moderate)
    """

    name = "elo"

    def __init__(self, k_factor: float = 20.0, home_advantage: float = 100.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: Dict[str, float] = {}

    def expected_prob(self, rating_a: float, rating_b: float,
                       home: bool = True) -> float:
        """Expected win probability for team A (home)."""
        ha = self.home_advantage if home else 0
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a - ha) / 400.0))

    def update_ratings(self, team_a: str, team_b: str,
                       team_a_won: bool, home: bool = True):
        """Update ELO ratings after a game."""
        rating_a = self.ratings.get(team_a, 1500.0)
        rating_b = self.ratings.get(team_b, 1500.0)

        expected = self.expected_prob(rating_a, rating_b, home)
        actual = 1.0 if team_a_won else 0.0

        new_rating_a = rating_a + self.k_factor * (actual - expected)
        new_rating_b = rating_b + self.k_factor * ((1.0 - actual) - (1.0 - expected))

        self.ratings[team_a] = new_rating_a
        self.ratings[team_b] = new_rating_b

    def fit_from_games(self, games_df: pd.DataFrame):
        """Train ELO ratings from historical game results."""
        for _, game in games_df.iterrows():
            home_team = game.get("TEAM_NAME_home", "")
            away_team = game.get("TEAM_NAME_away", "")

            # CRITICAL FIX: WL_home is a string ("W"/"L"), not 0/1.
            # The old code compared against int 1 which NEVER matched.
            wl_raw = game.get("WL_home", "")
            if isinstance(wl_raw, str):
                home_won = wl_raw.strip().upper() == "W"
            else:
                home_won = bool(game.get("point_diff", 0) > 0)

            if home_team and away_team:
                self.update_ratings(home_team, away_team, home_won, home=True)

    def predict(self, features: Dict) -> ModelPrediction:
        home_team = features.get("TEAM_NAME_home", "")
        away_team = features.get("TEAM_NAME_away", "")

        home_elo = self.ratings.get(home_team, 1500.0)
        away_elo = self.ratings.get(away_team, 1500.0)

        prob = self.expected_prob(home_elo, away_elo, home=True)

        # Confidence based on total games played by ALL teams (proxy for
        # how established the ratings are overall, not just per-team).
        # If we've seen < 10 games total, ratings are unreliable.
        n_ratings = len(self.ratings)
        base_confidence = min(0.3 + n_ratings * 0.01, 0.85)
        # Edge case: if neither team has played before, ELO knows nothing
        if home_elo == 1500.0 and away_elo == 1500.0:
            base_confidence = 0.15
        confidence = base_confidence

        return ModelPrediction(
            name="elo",
            probability=prob,
            confidence=confidence,
        )


class MarketModel(BaseModel):
    """
    Market-Based Model: extracts signal from market consensus.

    The market is efficient but not perfect. We can extract:
    - Consensus win probability (implied from ML)
    - Sharps vs public money (line movement direction)
    - Reverse line movement (RLM) signals

    This model returns the market-implied probability adjusted
    by detected market signals.
    """

    name = "market"

    def predict(self, features: Dict) -> ModelPrediction:
        implied_prob = features.get("market_home_win_prob",
                       features.get("implied_home_win_prob", 0.5))

        line_movement_signal = features.get("line_movement_signal", 0.0)
        sharp_signal = features.get("sharp_money_signal", 0.0)

        # Blend signals
        adjusted = implied_prob
        if line_movement_signal > 0:
            adjusted += 0.01 * min(line_movement_signal, 5)
        if sharp_signal > 0:
            adjusted += 0.005 * min(sharp_signal, 10)

        adjusted = max(0.01, min(0.99, adjusted))

        return ModelPrediction(
            name="market",
            probability=adjusted,
            confidence=0.7,  # Market is generally reliable
        )


class SklearnModel(BaseModel):
    """Wrapper for sklearn/XGBoost/LightGBM models."""

    def __init__(self, name: str, model_obj, feature_cols: List[str]):
        self.name = name
        self.model = model_obj
        self.feature_cols = feature_cols

    def predict(self, features: Dict) -> ModelPrediction:
        try:
            # Build feature vector
            X = np.array([[features.get(c, 0.0) for c in self.feature_cols]])

            if hasattr(self.model, "predict_proba"):
                prob = self.model.predict_proba(X)[0][1]
            else:
                prob = self.model.predict(X)[0]
                prob = max(0.01, min(0.99, float(prob)))

            prob = float(max(0.01, min(0.99, prob)))

            return ModelPrediction(
                name=self.name,
                probability=prob,
                confidence=0.8,
            )
        except Exception:
            return ModelPrediction(
                name=self.name,
                probability=0.5,
                confidence=0.3,
            )


# ═══════════════════════════════════════════════════════════════════════════
#  ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════

class EnsembleModel:
    """
    Weighted ensemble of multiple models with adaptive weights.

    Weights are learned from recent out-of-sample performance.
    The ensemble uses Brier score weighting: models that were more
    accurate recently get higher weight.

    Usage:
        ensemble = EnsembleModel()
        ensemble.add_model(elo_model)
        ensemble.add_model(market_model)

        prediction = ensemble.predict(features)
        print(prediction.home_win_probability)
        print(prediction.model_weights)
    """

    def __init__(self, default_weights: Optional[Dict[str, float]] = None):
        self.models: List[BaseModel] = []
        self.weights: Dict[str, float] = default_weights or {}
        self.recent_performance: Dict[str, List[float]] = {}
        self._n_predictions = 0

    def add_model(self, model: BaseModel, weight: Optional[float] = None):
        """Add a model to the ensemble."""
        self.models.append(model)
        if weight is not None:
            self.weights[model.name] = weight
        elif model.name not in self.weights:
            self.weights[model.name] = 1.0

    def remove_model(self, name: str):
        """Remove a model by name."""
        self.models = [m for m in self.models if m.name != name]
        self.weights.pop(name, None)

    def predict(self, features: Dict) -> EnsemblePrediction:
        """
        Run all models and produce ensemble prediction.

        Args:
            features: Feature dictionary for a single game

        Returns:
            EnsemblePrediction with probabilities and model breakdown
        """
        predictions = []
        valid_weights = []

        for model in self.models:
            try:
                pred = model.predict(features)
                weight = self.weights.get(model.name, 1.0)

                if pred.confidence > 0.3:  # Skip low-confidence predictions
                    predictions.append(pred)
                    valid_weights.append(weight)
            except Exception:
                continue

        if not predictions:
            return EnsemblePrediction(
                home_win_probability=0.5,
                num_models=0,
                model_name="ensemble",
                generated_at=datetime.now().isoformat(),
            )

        # Normalize weights
        total_weight = sum(valid_weights)
        normalized_weights = [w / total_weight for w in valid_weights]

        # Weighted ensemble
        home_prob = sum(p.probability * w
                        for p, w in zip(predictions, normalized_weights))

        # Store weights for display
        model_weights = {
            p.name: round(w, 4)
            for p, w in zip(predictions, normalized_weights)
        }

        # Consensus level (1 - normalized variance)
        probs = [p.probability for p in predictions]
        variance = np.var(probs) if len(probs) > 1 else 0.0
        consensus = max(0.0, 1.0 - variance * 20)

        self._n_predictions += 1

        return EnsemblePrediction(
            home_win_probability=round(home_prob, 4),
            model_predictions=predictions,
            model_weights=model_weights,
            consensus_level=round(consensus, 2),
            num_models=len(predictions),
            model_variance=round(variance, 6),
            model_name="ensemble",
            generated_at=datetime.now().isoformat(),
        )

    def predict_batch(self, feature_rows: List[Dict]) -> List[EnsemblePrediction]:
        """Predict for multiple games."""
        return [self.predict(f) for f in feature_rows]

    def update_weights_from_performance(
        self,
        predictions: List[EnsemblePrediction],
        actual_outcomes: List[bool],
        window: int = 50,
        decay: float = 0.95,
    ):
        """
        Update model weights based on recent prediction accuracy.

        Uses Brier score (lower is better) with exponential decay
        giving more weight to recent performance.

        Args:
            predictions: List of past ensemble predictions
            actual_outcomes: List of actual outcomes (True = home win)
            window: How many recent games to consider
            decay: Exponential decay factor (0-1)
        """
        model_briers: Dict[str, List[float]] = {}

        for pred, actual in zip(predictions[-window:], actual_outcomes[-window:]):
            for mp in pred.model_predictions:
                if mp.name not in model_briers:
                    model_briers[mp.name] = []
                # Brier score component for this prediction
                brier = (mp.probability - (1.0 if actual else 0.0)) ** 2
                model_briers[mp.name].append(brier)

        # Compute weighted average Brier score
        for name, briers in model_briers.items():
            if not briers:
                continue
            weights = [decay ** i for i in range(len(briers))]
            weights = [w / sum(weights) for w in weights]
            avg_brier = sum(b * w for b, w in zip(briers, weights))

            # Convert Brier score to weight: lower Brier = higher weight
            # weight = 1 / (1 + brier * 10)
            new_weight = 1.0 / (1.0 + avg_brier * 10.0)
            self.weights[name] = round(new_weight, 4)

    def get_prediction_breakdown(self, prediction: EnsemblePrediction) -> str:
        """Format ensemble prediction breakdown for display."""
        lines = [
            f"🎯 ENSEMBLE PREDICTION",
            f"{'─' * 50}",
            f"Home Win: {prediction.home_win_probability:.1%}",
            f"Consensus: {prediction.consensus_level:.0%} | "
            f"Models: {prediction.num_models} | "
            f"Variance: {prediction.model_variance:.4f}",
        ]

        if prediction.model_predictions:
            lines.append(f"\nModel Breakdown:")
            for mp in sorted(prediction.model_predictions,
                            key=lambda x: prediction.model_weights.get(x.name, 0),
                            reverse=True):
                w = prediction.model_weights.get(mp.name, 1.0)
                lines.append(
                    f"  {mp.name:12s}: {mp.probability:.1%} "
                    f"(weight: {w:.2f})"
                )

        if prediction.predicted_total:
            lines.append(f"\nPredicted Total: {prediction.predicted_total:.1f}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def build_ensemble(
    elo_ratings: Optional[Dict[str, float]] = None,
    sklearn_models: Optional[List[Tuple[str, Any, List[str]]]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> EnsembleModel:
    """
    Build a pre-configured ensemble with all available models.

    Args:
        elo_ratings: Pre-computed ELO ratings (team_name -> rating)
        sklearn_models: List of (name, model_object, feature_cols) tuples
        weights: Optional custom weights

    Returns:
        Configured EnsembleModel
    """
    ensemble = EnsembleModel(default_weights=weights)

    # ELO model
    elo = ELOModel()
    if elo_ratings:
        elo.ratings = elo_ratings
    ensemble.add_model(elo)

    # Market model
    market = MarketModel()
    ensemble.add_model(market)

    # Sklearn models
    if sklearn_models:
        for name, model_obj, feature_cols in sklearn_models:
            wrapper = SklearnModel(name, model_obj, feature_cols)
            ensemble.add_model(wrapper)

    return ensemble
