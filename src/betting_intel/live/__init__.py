"""
Live Prediction Engine — real-time predictions for live + upcoming games.

This package provides a zero-synthetic-data prediction system that:
  1. Polls TheOddsAPI for real games (today + tomorrow)
  2. Detects live/in-progress games vs upcoming games
  3. Generates ML-powered predictions for each game using real market odds
  4. Auto-refreshes on a configurable interval
  5. Returns only real games — never generates fake/synthetic matchups

Usage:
    from betting_intel.live.engine import LivePredictionEngine

    engine = LivePredictionEngine()
    live_games = engine.get_live_games()        # Games in progress
    today_games = engine.get_today_games()       # Today's full card
    two_day_window = engine.get_next_two_days()  # Today + Tomorrow
"""

from betting_intel.live.engine import LivePredictionEngine

__all__ = ["LivePredictionEngine"]
