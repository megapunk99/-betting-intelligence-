"""
Backtesting engine with walk-forward validation.
Simulates how a strategy would have performed in real time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from config import (
    WALK_FORWARD_WINDOW, WALK_FORWARD_STEP,
    MIN_TRAIN_SAMPLES, INITIAL_BANKROLL,
    UNIT_SIZE, MIN_EDGE_THRESHOLD
)


@dataclass
class BacktestResult:
    """Container for backtest results."""

    strategy_name: str
    model_name: str
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0
    roi: float = 0.0
    kelly_roi: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_odds: float = 0.0
    avg_edge: float = 0.0
    profit_by_date: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    bets_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    model_metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class WalkForwardEngine:
    """
    Walk-forward backtesting: trains on rolling window, tests on next period.
    This is the gold standard for time-series validation in betting.
    Prevents lookahead bias and simulates real-world performance.
    """

    def __init__(
        self,
        train_window: int = WALK_FORWARD_WINDOW,
        step: int = WALK_FORWARD_STEP,
        min_train: int = MIN_TRAIN_SAMPLES,
    ):
        self.train_window = train_window
        self.step = step
        self.min_train = min_train
        self.results: List[BacktestResult] = []

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        model_builder: Callable,
        strategy_name: str,
        model_name: str,
        prediction_type: str = "regression",
        make_bets: bool = True,
        market_odds_col: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
            model_builder: Function that returns a fresh model instance
            strategy_name: Name of the strategy
            model_name: Name of the model
            prediction_type: 'regression' or 'classification'
            make_bets: Whether to generate betting decisions
            market_odds_col: Column with market implied odds (if available)
        """
        result = BacktestResult(
            strategy_name=strategy_name,
            model_name=model_name
        )

        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        n = len(df)

        if n < self.min_train:
            result.errors.append(f"Not enough data: {n} < {self.min_train}")
            return result

        # Walk-forward loop
        predictions = []
        actuals = []
        dates = []
        bet_records = []
        train_scores = []

        start_idx = 0
        while start_idx + self.train_window < n:
            train_end = start_idx + self.train_window
            test_end = min(train_end + self.step, n)

            # Split data
            train_df = df.iloc[start_idx:train_end]
            test_df = df.iloc[train_end:test_end]

            if len(train_df) < self.min_train:
                start_idx += self.step
                continue

            # Train
            X_train = train_df[feature_cols].dropna()
            y_train = train_df.loc[X_train.index, target_col]

            if len(X_train) < 50:
                start_idx += self.step
                continue

            model = model_builder()
            try:
                model.fit(X_train.values, y_train.values)
            except Exception as e:
                result.errors.append(f"Fit error at window {start_idx}: {str(e)}")
                start_idx += self.step
                continue

            # Test
            X_test = test_df[feature_cols].dropna()
            if len(X_test) == 0:
                start_idx += self.step
                continue

            test_indices = X_test.index
            try:
                y_pred = model.predict(X_test.values)
                y_actual = test_df.loc[test_indices, target_col].values

                predictions.extend(y_pred.tolist())
                actuals.extend(y_actual.tolist())
                dates.extend(test_df.loc[test_indices, "GAME_DATE"].tolist())

                # Store bet records
                if make_bets:
                    for i, idx in enumerate(test_indices):
                        row = df.loc[idx]
                        pred = y_pred[i]
                        actual = y_actual[i]

                        bet = self._create_bet_record(
                            row, pred, actual, prediction_type,
                            strategy_name, model_name,
                            feature_cols
                        )
                        if bet:
                            bet_records.append(bet)

                # Track training metrics
                if prediction_type == "regression":
                    y_train_pred = model.predict(X_train.values)
                    from sklearn.metrics import r2_score
                    train_r2 = r2_score(y_train.values, y_train_pred)
                    train_scores.append(train_r2)

            except Exception as e:
                result.errors.append(f"Predict error at window {start_idx}: {str(e)}")
                continue

            start_idx += self.step

        if len(predictions) == 0:
            result.errors.append("No predictions generated")
            return result

        # Compile results
        result.total_bets = len(bet_records)
        if bet_records:
            result.bets_df = pd.DataFrame(bet_records)
            self._compute_performance(result)

        # Model metrics
        if prediction_type == "regression" and len(predictions) > 0:
            from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
            result.model_metrics = {
                "mae": mean_absolute_error(actuals, predictions),
                "rmse": np.sqrt(mean_squared_error(actuals, predictions)),
                "r2": r2_score(actuals, predictions),
                "n_predictions": len(predictions),
                "avg_train_r2": np.mean(train_scores) if train_scores else 0,
            }

        return result

    def _create_bet_record(
        self,
        row: pd.Series,
        prediction: float,
        actual: float,
        prediction_type: str,
        strategy_name: str,
        model_name: str,
        feature_cols: List[str],
    ) -> Optional[Dict]:
        """Create a bet record from a prediction."""
        try:
            if prediction_type == "regression":
                # Total points betting
                actual_total = row.get("total_points", actual)
                predicted_total = prediction

                # Market line baseline: use trailing average as a proxy for the sportsbook's line
                # This is deliberately computed from lagged data only, and is NOT a feature
                # the model sees during training (it's excluded in select_features()).
                market_line = row.get(
                    "market_line_baseline",
                    row.get("trailing_avg_total_10g", predicted_total)
                )

                # Edge = our prediction vs market
                edge_pct = (predicted_total - market_line) / market_line if market_line > 0 else 0.0

                if abs(edge_pct) < MIN_EDGE_THRESHOLD:
                    return None

                # Determine bet side
                if edge_pct > 0:
                    bet_side = "OVER"
                    predicted_market = market_line
                else:
                    bet_side = "UNDER"
                    predicted_market = market_line

                # Win/loss
                if (bet_side == "OVER" and actual_total > predicted_market) or \
                   (bet_side == "UNDER" and actual_total < predicted_market):
                    outcome = "WIN"
                    profit = 1.0  # 1 unit profit at -110 odds
                elif actual_total == predicted_market:
                    outcome = "PUSH"
                    profit = 0.0
                else:
                    outcome = "LOSS"
                    profit = -1.0  # 1 unit loss at -110 odds

                return {
                    "game_date": row["GAME_DATE"],
                    "game_id": row["GAME_ID"],
                    "matchup": f"{row.get('TEAM_NAME_home', '?')} vs {row.get('TEAM_NAME_away', '?')}",
                    "strategy": strategy_name,
                    "model": model_name,
                    "bet_type": f"TOTAL_{bet_side}",
                    "predicted_total": float(predicted_total),
                    "market_line": float(market_line),
                    "actual_total": float(actual_total),
                    "edge_pct": float(edge_pct),
                    "outcome": outcome,
                    "profit_units": profit,
                }

            elif prediction_type == "classification":
                # Spread/moneyline betting
                pred_class = prediction
                actual_class = 1 if actual > 0 else 0

                if pred_class == actual_class:
                    outcome = "WIN"
                    profit = 1.0
                else:
                    outcome = "LOSS"
                    profit = -1.0

                return {
                    "game_date": row["GAME_DATE"],
                    "game_id": row["GAME_ID"],
                    "matchup": f"{row.get('TEAM_NAME_home', '?')} vs {row.get('TEAM_NAME_away', '?')}",
                    "strategy": strategy_name,
                    "model": model_name,
                    "bet_type": "SPREAD",
                    "predicted_class": int(pred_class),
                    "actual_class": int(actual_class),
                    "outcome": outcome,
                    "profit_units": profit,
                }

        except Exception as e:
            return None

        return None

    def _compute_performance(self, result: BacktestResult):
        """Compute performance metrics from bet records."""
        df = result.bets_df
        if len(df) == 0:
            return

        result.total_bets = len(df)
        result.wins = len(df[df["outcome"] == "WIN"])
        result.losses = len(df[df["outcome"] == "LOSS"])
        result.pushes = len(df[df["outcome"] == "PUSH"])
        total_decided = result.wins + result.losses

        if total_decided > 0:
            result.win_rate = result.wins / total_decided

        result.total_profit = df["profit_units"].sum()
        result.roi = result.total_profit / max(result.total_bets, 1) * 100

        # Cumulative profit (for drawdown calculation)
        df_sorted = df.sort_values("game_date")
        cumulative = df_sorted["profit_units"].cumsum()
        result.profit_by_date = pd.Series(
            cumulative.values,
            index=pd.to_datetime(df_sorted["game_date"].values)
        )

        # Max drawdown
        if len(cumulative) > 0:
            running_max = cumulative.cummax()
            drawdown = cumulative - running_max
            result.max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        # Sharpe ratio (assuming risk-free rate = 0)
        if len(df_sorted) > 5:
            result.sharpe_ratio = (
                df_sorted["profit_units"].mean()
                / max(df_sorted["profit_units"].std(), 1e-6)
                * np.sqrt(82)  # Approximate games per season
            )

        result.avg_edge = df["edge_pct"].mean() if "edge_pct" in df.columns else 0.0

    def summary(self) -> pd.DataFrame:
        """Generate summary DataFrame of all backtest results."""
        rows = []
        for r in self.results:
            rows.append({
                "Strategy": r.strategy_name,
                "Model": r.model_name,
                "Bets": r.total_bets,
                "Wins": r.wins,
                "Losses": r.losses,
                "Win Rate": f"{r.win_rate:.1%}",
                "Profit (units)": f"{r.total_profit:.1f}",
                "ROI": f"{r.roi:.1f}%",
                "Max DD": f"{r.max_drawdown:.1f}",
                "Sharpe": f"{r.sharpe_ratio:.2f}",
                "Errors": len(r.errors),
            })
        return pd.DataFrame(rows)
