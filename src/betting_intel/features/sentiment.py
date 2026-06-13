"""
Social Sentiment Analysis — NLP-based sentiment extraction from news and social media.

INSIGHT: Social media sentiment correlates with market inefficiencies.
  - Positive sentiment on a team → public bets that team → inflated lines
  - Negative sentiment → value on the other side
  - Sudden sentiment shifts (injuries, scandals, hot streaks) create temporary edges

DATA SOURCE REQUIREMENTS:
  - Twitter/X API (paid, most timely)
  - News API / Google News RSS (free tier available)
  - Reddit API (free, /r/nba sentiment)
  - ESPN comments / article sentiment

This module provides:
  1. SentimentData model — structure for storing sentiment scores
  2. SentimentFetcher — abstract base to fetch from sources
  3. Stub implementation returning neutral values

USAGE:
    from betting_intel.features.sentiment import get_sentiment_features
    features = get_sentiment_features(home_team, away_team)
    # Returns zero-vector (stub) until data sources are connected
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SentimentData:
    """Sentiment analysis results for a team or game.
    
    All scores are -1.0 (extremely negative) to +1.0 (extremely positive).
    """
    def __init__(
        self,
        team_name: str = "",
        overall_sentiment: float = 0.0,
        volume: int = 0,             # Number of mentions/documents
        sentiment_std: float = 0.0,   # Volatility of sentiment
        source: str = "stub",
    ):
        self.team_name = team_name
        self.overall_sentiment = overall_sentiment
        self.volume = volume
        self.sentiment_std = sentiment_std
        self.source = source


class SentimentFetcher:
    """Fetch sentiment data from external sources.
    
    Currently a stub — returns neutral values until a data source
    is connected.
    """
    
    def __init__(self):
        self._cache: dict[str, SentimentData] = {}
    
    def fetch_team_sentiment(self, team_name: str) -> SentimentData:
        """Fetch social/news sentiment for a team.
        
        Stub — returns neutral (0.0) sentiment.
        """
        return SentimentData(team_name=team_name, source="stub")
    
    def clear_cache(self) -> None:
        self._cache.clear()


def get_sentiment_features(
    home_team: str,
    away_team: str,
) -> dict[str, float]:
    """One-shot: get sentiment features for a game.
    
    Returns zero-vector (stub) until data source is connected.
    """
    return {
        "sent_home_overall": 0.0,
        "sent_away_overall": 0.0,
        "sent_diff": 0.0,
        "sent_home_volume": 0.0,
        "sent_away_volume": 0.0,
        "sent_volume_diff": 0.0,
    }


__all__ = [
    "SentimentData", "SentimentFetcher", "get_sentiment_features",
]
