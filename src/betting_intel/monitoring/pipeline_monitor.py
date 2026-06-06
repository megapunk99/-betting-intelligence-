"""
Pipeline Monitor — tracks pipeline runs, timing, and status.

Inspired by NBA_AI's PipelineMonitor. Provides:
  - Run tracking with unique IDs
  - Per-stage timing and status
  - Performance metrics over time
  - Error and warning aggregation

Usage:
    monitor = PipelineMonitor()
    run_id = monitor.start_run("live", "2025-24")
    # ... pipeline stages execute ...
    monitor.complete_run(run_id, status="success", games=10, predictions=15)
    history = monitor.get_recent_runs(limit=10)
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineRun:
    """A single pipeline execution record."""
    run_id: str
    mode: str  # "live", "historical", "full", "post-game", "pre-game"
    season: str
    status: str = "running"  # "running", "success", "partial", "failed"
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    games_processed: int = 0
    predictions_generated: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


class PipelineMonitor:
    """
    Tracks pipeline runs with timing, status, and performance history.

    Stores run history in a JSON file for persistence across executions.
    Provides performance metrics over time for trend analysis.

    Usage:
        monitor = PipelineMonitor()
        run_id = monitor.start_run("live", "2025-24")
        # ... run pipeline stages ...
        monitor.complete_run(run_id, status="success", games=10,
                             predictions=15)

        # View history
        for run in monitor.get_recent_runs(5):
            print(f"{run.mode}: {run.status} ({run.duration_seconds:.1f}s)")
    """

    def __init__(self, history_path: Optional[Path] = None):
        self.history_path = history_path or Path("data/pipeline_history.json")
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._current_run: Optional[PipelineRun] = None
        self._runs: List[PipelineRun] = self._load_history()

    def _load_history(self) -> List[PipelineRun]:
        """Load run history from disk."""
        if self.history_path.exists():
            try:
                with open(self.history_path, "r") as f:
                    data = json.load(f)
                return [PipelineRun(**r) for r in data]
            except Exception as e:
                logger.warning(f"Failed to load pipeline history: {e}")
        return []

    def _save_history(self):
        """Save run history to disk."""
        try:
            data = [asdict(r) for r in self._runs[-100:]]  # Keep last 100 runs
            with open(self.history_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save pipeline history: {e}")

    def start_run(self, mode: str, season: str = "Current") -> str:
        """
        Start tracking a new pipeline run.

        Args:
            mode: Pipeline mode ("live", "historical", "full", etc.)
            season: NBA season identifier

        Returns:
            run_id: Unique identifier for this run
        """
        run_id = f"{mode}_{datetime.now():%Y%m%d_%H%M%S}_{len(self._runs)}"
        self._current_run = PipelineRun(
            run_id=run_id,
            mode=mode,
            season=season,
            status="running",
            started_at=datetime.now().isoformat(),
        )
        self._runs.append(self._current_run)
        self._save_history()
        logger.info(f"Pipeline run started: {run_id} ({mode})")
        return run_id

    def complete_run(self, run_id: str, status: str = "success",
                     games: int = 0, predictions: int = 0,
                     errors: Optional[List[str]] = None,
                     warnings: Optional[List[str]] = None,
                     metrics: Optional[Dict[str, float]] = None):
        """
        Mark a pipeline run as complete.

        Args:
            run_id: Run ID from start_run()
            status: "success", "partial", or "failed"
            games: Number of games processed
            predictions: Number of predictions generated
            errors: List of error messages
            warnings: List of warning messages
            metrics: Additional performance metrics
        """
        run = self._find_run(run_id)
        if run is None:
            logger.warning(f"Run {run_id} not found")
            return

        run.status = status
        run.completed_at = datetime.now().isoformat()
        run.duration_seconds = time.time() - (
            datetime.fromisoformat(run.started_at).timestamp()
            if run.started_at else time.time()
        )
        run.games_processed = games
        run.predictions_generated = predictions
        if errors:
            run.errors = errors
        if warnings:
            run.warnings = warnings
        if metrics:
            run.metrics = metrics

        self._save_history()
        logger.info(
            f"Pipeline run {status}: {run_id} "
            f"({run.duration_seconds:.1f}s, {games} games, "
            f"{predictions} predictions, {len(errors or [])} errors)"
        )

    def record_stage(self, run_id: str, stage_name: str,
                     status: str = "ok", duration: float = 0.0,
                     **details):
        """Record a pipeline stage result.

        Args:
            run_id: Run ID from start_run()
            stage_name: Name of the pipeline stage
            status: Stage status ("ok", "error", "warning", "skipped")
            duration: Stage duration in seconds
            **details: Additional stage-specific data
        """
        run = self._find_run(run_id)
        if run is None:
            return
        run.stages[stage_name] = {
            "status": status,
            "duration": round(duration, 2),
            **details,
        }
        self._save_history()

    def get_recent_runs(self, limit: int = 10,
                        mode: Optional[str] = None) -> List[PipelineRun]:
        """Get the most recent pipeline runs.

        Args:
            limit: Max number of runs to return
            mode: Optional filter by mode ("live", "historical", etc.)

        Returns:
            List of PipelineRun objects
        """
        runs = self._runs
        if mode:
            runs = [r for r in runs if r.mode == mode]
        return runs[-limit:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get aggregate performance metrics across recent runs.

        Returns:
            Dict with total_runs, success_rate, avg_duration, etc.
        """
        recent = self._runs[-50:]
        if not recent:
            return {"total_runs": 0}

        total = len(recent)
        successes = sum(1 for r in recent if r.status == "success")
        partials = sum(1 for r in recent if r.status == "partial")
        failures = sum(1 for r in recent if r.status == "failed")

        avg_duration = sum(r.duration_seconds for r in recent) / total
        total_games = sum(r.games_processed for r in recent)
        total_preds = sum(r.predictions_generated for r in recent)
        total_errors = sum(len(r.errors) for r in recent)

        return {
            "total_runs": total,
            "success_rate": successes / total if total > 0 else 0,
            "partial_rate": partials / total if total > 0 else 0,
            "failure_rate": failures / total if total > 0 else 0,
            "avg_duration_seconds": round(avg_duration, 1),
            "total_games_processed": total_games,
            "total_predictions": total_preds,
            "total_errors": total_errors,
            "period": f"last {total} runs",
        }

    def get_model_performance(self, model_name: str) -> Dict[str, float]:
        """Get performance metrics for a specific model across runs.

        Args:
            model_name: Name of the model to look up

        Returns:
            Dict with performance metrics or empty dict if no data
        """
        # This can be extended to track per-model metrics from run data
        return {
            "runs_included": len([r for r in self._runs
                                   if model_name in str(r.stages)]),
        }

    def _find_run(self, run_id: str) -> Optional[PipelineRun]:
        """Find a run by ID."""
        for run in self._runs:
            if run.run_id == run_id:
                return run
        return None


# ── Model Performance Tracker (ATS Evaluation) ──────────────────────────


@dataclass
class ATSRecord:
    """A single against-the-spread prediction record."""
    game_id: str
    game_date: str
    matchup: str
    model_name: str
    predicted_spread: float
    vegas_spread: float
    actual_margin: float
    home_covered: bool  # True if home team covered the spread
    model_correct: bool  # True if model correctly predicted the side


class ATSTracker:
    """
    Tracks against-the-spread (ATS) performance for model evaluation.

    Like NBA_AI's dashboard: tracks each model's ATS record,
    win rate, and performance over time.

    Usage:
        tracker = ATSTracker()
        tracker.record_prediction(
            game_id="001", game_date="2025-01-15",
            matchup="LAL @ BOS",
            model_name="ensemble",
            predicted_spread=-4.5, vegas_spread=-5.5,
            actual_margin=3.0,
        )
        summary = tracker.get_summary()
        print(f"ATS: {summary['wins']}-{summary['losses']} "
              f"({summary['win_rate']:.1%})")
    """

    def __init__(self, history_path: Optional[Path] = None):
        self.history_path = history_path or Path("data/ats_history.json")
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: List[ATSRecord] = self._load_history()

    def _load_history(self) -> List[ATSRecord]:
        """Load ATS history from disk."""
        if self.history_path.exists():
            try:
                with open(self.history_path, "r") as f:
                    data = json.load(f)
                return [ATSRecord(**r) for r in data]
            except Exception as e:
                logger.warning(f"Failed to load ATS history: {e}")
        return []

    def _save_history(self):
        """Save ATS history to disk."""
        try:
            data = [
                {
                    "game_id": r.game_id,
                    "game_date": r.game_date,
                    "matchup": r.matchup,
                    "model_name": r.model_name,
                    "predicted_spread": r.predicted_spread,
                    "vegas_spread": r.vegas_spread,
                    "actual_margin": r.actual_margin,
                    "home_covered": r.home_covered,
                    "model_correct": r.model_correct,
                }
                for r in self._records[-5000:]
            ]
            with open(self.history_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save ATS history: {e}")

    def record_prediction(self, game_id: str, game_date: str, matchup: str,
                           model_name: str, predicted_spread: float,
                           vegas_spread: float, actual_margin: float):
        """Record a prediction for ATS evaluation.

        A team covers the spread if:
          - For home favorite (spread < 0): home_margin > -spread
          - For home underdog (spread > 0): home_margin > spread
        The model is correct if it predicts the same side as the outcome.
        """
        # Determine if home covered the Vegas spread
        # Home covers if actual_margin > -vegas_spread
        # e.g. spread=-5.5 (favored by 5.5): covers if actual_margin > 5.5
        # e.g. spread=+5.5 (underdog by 5.5): covers if actual_margin > -5.5
        home_covered = actual_margin > -vegas_spread

        # Determine if model is correct (same side as actual outcome)
        # Model predicts home covers if predicted_spread > vegas_spread
        # (model thinks home will outperform the spread)
        model_thinks_home_covers = predicted_spread > vegas_spread
        model_correct = model_thinks_home_covers == home_covered

        record = ATSRecord(
            game_id=game_id,
            game_date=game_date,
            matchup=matchup,
            model_name=model_name,
            predicted_spread=predicted_spread,
            vegas_spread=vegas_spread,
            actual_margin=actual_margin,
            home_covered=home_covered,
            model_correct=model_correct,
        )
        self._records.append(record)
        self._save_history()

    def get_summary(self, model_name: Optional[str] = None,
                    last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get ATS performance summary.

        Args:
            model_name: Optional filter by model
            last_n: Optional limit to last N predictions

        Returns:
            Dict with wins, losses, win_rate, push_rate, etc.
        """
        records = self._records
        if model_name:
            records = [r for r in records if r.model_name == model_name]
        if last_n:
            records = records[-last_n:]

        if not records:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0}

        wins = sum(1 for r in records if r.model_correct)
        total = len(records)
        win_rate = wins / total if total > 0 else 0.0

        # Recent trend (last 20)
        recent = records[-20:]
        recent_wins = sum(1 for r in recent if r.model_correct)
        recent_rate = recent_wins / len(recent) if recent else 0.0

        return {
            "total": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round(win_rate, 4),
            "recent_win_rate": round(recent_rate, 4),
            "avg_margin_error": round(
                sum(abs(r.predicted_spread - r.actual_margin)
                    for r in records) / total, 2
            ),
        }

    def get_model_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Compare ATS performance across all models.

        Returns:
            Dict mapping model_name -> performance summary
        """
        models = set(r.model_name for r in self._records)
        comparison = {}
        for model in models:
            comparison[model] = self.get_summary(model_name=model)
        return comparison
