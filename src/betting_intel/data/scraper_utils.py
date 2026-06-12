"""
Scraper Utilities — shared retry, backoff, and health tracking for ALL data sources.

Every scraper in the system uses these utilities so that:
  1. Failures are handled consistently (exponential backoff, jitter)
  2. Health metrics are tracked uniformly (success rate, latency, error types)
  3. The ScraperCoordinator can monitor all scrapers from a single point

Usage:
    from betting_intel.data.scraper_utils import retry_with_backoff, ScraperHealthMonitor

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def fetch_odds():
        # ... fragile network call ...
        pass

    monitor = ScraperHealthMonitor()
    monitor.record_success("espn_http", latency_ms=450)
    monitor.record_failure("espn_http", "HTTP_503")
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

# ── Type ────────────────────────────────────────────────────────────────────

F = TypeVar("F", bound=Callable[..., Any])

# ── Retry with Exponential Backoff + Jitter ─────────────────────────────────


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        OSError,
        IOError,
    ),
) -> Callable[[F], F]:
    """
    Decorator: retry a function with exponential backoff + jitter.

    Args:
        max_retries: Maximum number of retry attempts (default 3).
        base_delay: Initial delay in seconds (default 1.0).
        max_delay: Maximum delay cap in seconds (default 30.0).
        backoff_factor: Multiplier applied each retry (default 2.0).
        jitter: Random jitter fraction applied to delay (default 0.1 = ±10%).
        retryable_exceptions: Tuple of exception types that trigger a retry.

    Example:
        @retry_with_backoff(max_retries=3, base_delay=0.5)
        def fetch_espn_scoreboard():
            # ... fragile HTTP call ...
            pass
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        # Add jitter: ±jitter% of the current delay
                        jitter_amount = delay * jitter * (2 * random.random() - 1)
                        actual_delay = min(delay + jitter_amount, max_delay)
                        logger.debug(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} "
                            f"failed: {e}. Retrying in {actual_delay:.1f}s..."
                        )
                        time.sleep(actual_delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.warning(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                except Exception as e:
                    # Non-retryable exceptions propagate immediately
                    raise

            raise last_exception  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


def retry_with_backoff_async(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        OSError,
        IOError,
    ),
) -> Callable[[F], F]:
    """
    Async version of retry_with_backoff.

    Usage:
        @retry_with_backoff_async(max_retries=3)
        async def fetch_odds():
            ...
    """
    import asyncio

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        jitter_amount = delay * jitter * (2 * random.random() - 1)
                        actual_delay = min(delay + jitter_amount, max_delay)
                        logger.debug(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} "
                            f"failed: {e}. Retrying in {actual_delay:.1f}s..."
                        )
                        await asyncio.sleep(actual_delay)
                        delay = min(delay * backoff_factor, max_delay)
                except Exception as e:
                    raise

            raise last_exception  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


# ── Scraper Health Monitor ─────────────────────────────────────────────────


class ScraperHealthRecord:
    """Health metrics for a single scraper over a time window."""

    def __init__(self, scraper_name: str, window_minutes: int = 60):
        self.scraper_name = scraper_name
        self.window_minutes = window_minutes
        self._calls: list[dict] = []

    def record_success(self, latency_ms: float, source: str = ""):
        """Record a successful scrape call."""
        self._calls.append({
            "status": "success",
            "latency_ms": latency_ms,
            "source": source,
            "timestamp": time.time(),
        })
        self._prune()

    def record_failure(self, error_type: str, source: str = "", latency_ms: float = 0):
        """Record a failed scrape call."""
        self._calls.append({
            "status": "failure",
            "error_type": error_type,
            "source": source,
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        })
        self._prune()

    def _prune(self):
        """Remove records older than the window."""
        cutoff = time.time() - (self.window_minutes * 60)
        self._calls = [c for c in self._calls if c["timestamp"] >= cutoff]

    @property
    def total_calls(self) -> int:
        return len(self._calls)

    @property
    def successes(self) -> int:
        return sum(1 for c in self._calls if c["status"] == "success")

    @property
    def failures(self) -> int:
        return sum(1 for c in self._calls if c["status"] == "failure")

    @property
    def success_rate(self) -> float:
        if not self._calls:
            return 1.0  # No data = assume healthy
        return self.successes / len(self._calls)

    @property
    def avg_latency_ms(self) -> float:
        latencies = [c["latency_ms"] for c in self._calls if c["status"] == "success"]
        return sum(latencies) / len(latencies) if latencies else 0.0

    @property
    def error_types(self) -> dict[str, int]:
        """Count of each error type."""
        counts: dict[str, int] = {}
        for c in self._calls:
            if c["status"] == "failure":
                err = c.get("error_type", "UNKNOWN")
                counts[err] = counts.get(err, 0) + 1
        return counts

    @property
    def error_rate(self) -> float:
        return 1.0 - self.success_rate

    def summary(self) -> dict:
        return {
            "scraper": self.scraper_name,
            "total_calls": self.total_calls,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(self.success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "error_types": self.error_types,
            "window_minutes": self.window_minutes,
            "healthy": self.success_rate >= 0.8,
        }


class ScraperHealthMonitor:
    """
    Central health monitor for ALL scrapers.

    Every scraper records successes and failures through this monitor,
    enabling:
      - Real-time health dashboards
      - Auto-disable of consistently failing scrapers
      - Per-source performance tracking

    Usage:
        monitor = ScraperHealthMonitor()
        monitor.record_success("draftkings", latency_ms=3200)
        monitor.record_failure("espn_http", "HTTP_429")
        status = monitor.get_health_summary()
    """

    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self._records: dict[str, ScraperHealthRecord] = {}
        self._disabled_until: dict[str, float] = {}

    def get_record(self, scraper_name: str) -> ScraperHealthRecord:
        """Get or create a health record for a scraper."""
        if scraper_name not in self._records:
            self._records[scraper_name] = ScraperHealthRecord(
                scraper_name, window_minutes=self.window_minutes,
            )
        return self._records[scraper_name]

    def record_success(self, scraper_name: str, latency_ms: float, source: str = ""):
        """Record a successful scrape."""
        self.get_record(scraper_name).record_success(latency_ms, source=source)
        # If it was disabled, re-enable it on success
        self._disabled_until.pop(scraper_name, None)

    def record_failure(
        self, scraper_name: str, error_type: str,
        source: str = "", latency_ms: float = 0,
    ):
        """Record a failed scrape."""
        self.get_record(scraper_name).record_failure(
            error_type, source=source, latency_ms=latency_ms,
        )

        # Auto-disable if failure rate > 50% in window and > 10 total calls
        record = self._records[scraper_name]
        if record.total_calls >= 10 and record.success_rate < 0.5:
            disable_minutes = min(30, 5 * (record.failures // 5))
            self._disabled_until[scraper_name] = time.time() + (disable_minutes * 60)
            logger.warning(
                f"Auto-disabled {scraper_name} for {disable_minutes}min "
                f"(success rate: {record.success_rate:.0%}, "
                f"{record.failures}/{record.total_calls} failures)"
            )

    def is_disabled(self, scraper_name: str) -> bool:
        """Check if a scraper is temporarily disabled due to high failure rate."""
        until = self._disabled_until.get(scraper_name)
        if until is None:
            return False
        if time.time() >= until:
            del self._disabled_until[scraper_name]
            logger.info(f"Re-enabled {scraper_name} after cooldown")
            return False
        return True

    def get_health_summary(self) -> dict[str, Any]:
        """Get health summary for all scrapers."""
        summaries = {}
        all_healthy = True
        for name, record in self._records.items():
            s = record.summary()
            s["disabled"] = self.is_disabled(name)
            s["disabled_remaining_seconds"] = max(
                0, self._disabled_until.get(name, 0) - time.time()
            )
            summaries[name] = s
            if not s["healthy"]:
                all_healthy = False

        return {
            "scrapers": summaries,
            "total_scrapers": len(self._records),
            "disabled_scrapers": sum(1 for n in self._records if self.is_disabled(n)),
            "all_healthy": all_healthy,
            "window_minutes": self.window_minutes,
            "generated_at": datetime.now().isoformat(),
        }

    def get_slowest_scraper(self) -> Optional[str]:
        """Get the scraper with the highest average latency."""
        best_name = None
        best_latency = -1.0
        for name, record in self._records.items():
            if record.avg_latency_ms > best_latency and record.total_calls > 0:
                best_latency = record.avg_latency_ms
                best_name = name
        return best_name

    def get_most_error_prone(self) -> Optional[str]:
        """Get the scraper with the highest error rate."""
        best_name = None
        best_error_rate = -1.0
        for name, record in self._records.items():
            if record.error_rate > best_error_rate and record.total_calls > 5:
                best_error_rate = record.error_rate
                best_name = name
        return best_name


# ── Global health monitor instance ─────────────────────────────────────────

# Single shared instance used by all scrapers
GLOBAL_SCRAPER_MONITOR = ScraperHealthMonitor()


# ── Source freshness tracker ────────────────────────────────────────────────


class SourceFreshnessTracker:
    """
    Tracks how fresh each data source is (when was it last successfully scraped).

    Used by the ScraperCoordinator and LiveDataGateway to determine
    whether cached data is still fresh enough to use.

    Usage:
        freshness = SourceFreshnessTracker()
        freshness.record_fetch("espn_http", "success")
        age = freshness.age_seconds("espn_http")
        if age > 120:
            print("ESPN HTTP data is stale (>2 min old)")
    """

    def __init__(self):
        self._last_fetch: dict[str, float] = {}
        self._last_status: dict[str, str] = {}

    def record_fetch(self, source: str, status: str = "success"):
        """Record that a source was fetched."""
        self._last_fetch[source] = time.time()
        self._last_status[source] = status

    def age_seconds(self, source: str) -> Optional[float]:
        """Get age of last fetch in seconds. None if never fetched."""
        ts = self._last_fetch.get(source)
        if ts is None:
            return None
        return time.time() - ts

    def is_fresh(self, source: str, max_age_seconds: float = 120) -> bool:
        """Check if a source's data is still fresh."""
        age = self.age_seconds(source)
        if age is None:
            return False
        return age < max_age_seconds

    def last_status(self, source: str) -> Optional[str]:
        """Get the status of the last fetch for a source."""
        return self._last_status.get(source)

    def summary(self) -> dict:
        """Get freshness summary for all sources."""
        result = {}
        now = time.time()
        for source, ts in self._last_fetch.items():
            result[source] = {
                "age_seconds": round(now - ts, 1),
                "status": self._last_status.get(source, "unknown"),
            }
        return result