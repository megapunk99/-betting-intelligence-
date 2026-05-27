"""
Data integrity module: staleness detection, feature freshness scoring,
and data leakage validation.

Mirrors the module at src/betting_intel/data/integrity.py for the
root-level import path used by main.py and predict_tomorrow.py.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path


# ── Data Freshness ─────────────────────────────────────────────────────────


@dataclass
class DataSourceStatus:
    """Status of a single data source."""

    name: str
    path: Optional[Path]
    last_modified: Optional[datetime]
    age_hours: float
    staleness_threshold_hours: float
    is_fresh: bool
    n_records: int = 0
    date_range: Optional[Tuple[date, date]] = None
    notes: List[str] = field(default_factory=list)


class DataFreshnessChecker:
    """
    Checks how fresh your data sources are.
    Detects stale databases, cache files, and missing recent data.
    """

    def __init__(self, staleness_thresholds: Optional[Dict[str, float]] = None):
        self.staleness_thresholds = staleness_thresholds or {
            "nba_db": 24.0,
            "cache": 1.0,
            "odds_api": 0.25,
            "model": 168.0,
            "output": 24.0,
        }

    def check_file(self, path: Path, source_type: str = "nba_db") -> DataSourceStatus:
        threshold = self.staleness_thresholds.get(source_type, 24.0)
        if not path.exists():
            return DataSourceStatus(
                name=path.name, path=path, last_modified=None,
                age_hours=float("inf"), staleness_threshold_hours=threshold,
                is_fresh=False, n_records=0, notes=["FILE_NOT_FOUND"],
            )
        stat = path.stat()
        last_modified = datetime.fromtimestamp(stat.st_mtime)
        age_hours = (datetime.now() - last_modified).total_seconds() / 3600
        is_fresh = age_hours <= threshold
        notes = []
        if not is_fresh:
            overdue_pct = (age_hours - threshold) / threshold * 100
            notes.append(f"STALE: {age_hours:.1f}h old (threshold: {threshold}h, {overdue_pct:.0f}% overdue)")
        return DataSourceStatus(
            name=path.name, path=path, last_modified=last_modified,
            age_hours=age_hours, staleness_threshold_hours=threshold,
            is_fresh=is_fresh, notes=notes,
        )

    def check_database(self, db_path: Path, table_names: List[str], date_column: str = "GAME_DATE") -> DataSourceStatus:
        import sqlite3
        threshold = self.staleness_thresholds.get("nba_db", 24.0)
        latest_date = None
        total_records = 0
        try:
            conn = sqlite3.connect(str(db_path))
            for table in table_names:
                try:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    n_records = cursor.fetchone()[0]
                    total_records += n_records
                    if n_records > 0:
                        cursor = conn.execute(f"SELECT MAX({date_column}) FROM {table}")
                        result = cursor.fetchone()[0]
                        if result:
                            date_val = pd.to_datetime(result).date()
                            if latest_date is None or date_val > latest_date:
                                latest_date = date_val
                except Exception:
                    continue
            conn.close()
        except Exception as e:
            return DataSourceStatus(
                name=db_path.name, path=db_path, last_modified=None,
                age_hours=float("inf"), staleness_threshold_hours=threshold,
                is_fresh=False, n_records=0, notes=[f"DB_ERROR: {str(e)}"],
            )
        if latest_date is None:
            return DataSourceStatus(
                name=db_path.name, path=db_path, last_modified=None,
                age_hours=float("inf"), staleness_threshold_hours=threshold,
                is_fresh=False, n_records=total_records, notes=["NO_DATA"],
            )
        days_since_update = (datetime.now().date() - latest_date).days
        is_fresh = days_since_update <= threshold / 24
        notes = []
        if not is_fresh:
            notes.append(f"STALE: Most recent data is {days_since_update} days old")
        return DataSourceStatus(
            name=db_path.name, path=db_path,
            last_modified=datetime.combine(latest_date, datetime.min.time()),
            age_hours=days_since_update * 24, staleness_threshold_hours=threshold,
            is_fresh=is_fresh, n_records=total_records,
            date_range=(date(2020, 1, 1), latest_date) if latest_date else None, notes=notes,
        )

    def get_freshness_score(self, statuses: List[DataSourceStatus]) -> Dict:
        if not statuses:
            return {"score": 0, "grade": "F", "details": "No data sources"}
        fresh_count = sum(1 for s in statuses if s.is_fresh)
        age_penalties = []
        for s in statuses:
            if s.age_hours > s.staleness_threshold_hours * 3:
                age_penalties.append(1.0)
            elif s.age_hours > s.staleness_threshold_hours:
                age_penalties.append((s.age_hours - s.staleness_threshold_hours) / (s.staleness_threshold_hours * 2))
            else:
                age_penalties.append(0.0)
        avg_penalty = np.mean(age_penalties) if age_penalties else 0
        score = max(0, fresh_count / max(len(statuses), 1) * 100 - avg_penalty * 50)
        grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 50 else "D" if score >= 25 else "F"
        return {"score": round(score, 1), "grade": grade,
                "fresh_sources": fresh_count, "total_sources": len(statuses),
                "freshness_ratio": round(fresh_count / max(len(statuses), 1), 2)}


# ── Leakage Validation ────────────────────────────────────────────────────


class LeakageValidator:
    """
    Validates that feature engineering does not introduce lookahead bias.
    Checks: rolling window shift(1), market line exclusion, temporal ordering, target leakage.
    """

    def __init__(self):
        self.issues: List[Dict] = []

    def check_market_line_leakage(self, feature_cols: List[str],
                                   market_cols: Optional[List[str]] = None) -> List[Dict]:
        if market_cols is None:
            market_cols = ["market_line_baseline", "market_line_pace_adj", "trailing_avg_total_10g"]
        issues = []
        for mc in market_cols:
            if mc in feature_cols:
                issues.append({
                    "severity": "critical", "feature": mc,
                    "issue": "MARKET LINE IN FEATURES",
                    "detail": f"'{mc}' should not be used as a model feature.",
                    "recommendation": f"Exclude '{mc}' from feature selection.",
                })
        return issues

    def check_temporal_ordering(self, df: pd.DataFrame, date_col: str = "GAME_DATE") -> List[Dict]:
        issues = []
        if date_col not in df.columns:
            issues.append({
                "severity": "critical", "feature": "GAME_DATE",
                "issue": "MISSING DATE COLUMN",
                "detail": "No date column found. Cannot validate temporal ordering.",
                "recommendation": "Add a date column to enable time-series validation.",
            })
            return issues
        dates = pd.to_datetime(df[date_col])
        if not dates.is_monotonic_increasing:
            issues.append({
                "severity": "warning", "feature": date_col,
                "issue": "DATA NOT SORTED CHRONOLOGICALLY",
                "recommendation": "Sort by date before feature engineering and backtesting.",
            })
        return issues

    def check_all(self, df: pd.DataFrame, feature_cols: List[str],
                   target_col: str = "total_points", date_col: str = "GAME_DATE") -> Dict:
        self.issues = []
        self.issues.extend(self.check_market_line_leakage(feature_cols))
        self.issues.extend(self.check_temporal_ordering(df, date_col))

        n_critical = sum(1 for i in self.issues if i["severity"] == "critical")
        n_warnings = sum(1 for i in self.issues if i["severity"] == "warning")
        return {
            "n_issues": len(self.issues), "critical": n_critical,
            "warnings": n_warnings, "passed": n_critical == 0,
            "issues": self.issues,
            "recommendation": "No leakage detected." if n_critical == 0
            else f"Fix {n_critical} critical issue(s) before using model predictions.",
        }


# ── Feature Freshness Scoring ──────────────────────────────────────────────


@dataclass
class FeatureFreshnessScore:
    feature_name: str
    source_type: str
    source_window: int
    recency: float
    score: float
    notes: List[str] = field(default_factory=list)


class FeatureFreshnessAnalyzer:
    def __init__(self):
        self.rolling_patterns = [
            (r"avg_pts_(\d+)g", "rolling_points"),
            (r"avg_pace_(\d+)g", "rolling_pace"),
            (r"avg_efg_(\d+)g", "rolling_efg"),
            (r"avg_pm_(\d+)g", "rolling_plus_minus"),
        ]

    def score_feature(self, feature_name: str, df: pd.DataFrame) -> FeatureFreshnessScore:
        static_patterns = ["IS_HOME", "home_", "team_", "GAME_", "SEASON_", "TEAM_", "rest_", "is_", "has_"]
        if any(feature_name.startswith(p) for p in static_patterns) and \
           not any(kw in feature_name for kw in ["rolling", "avg_", "cum_"]):
            return FeatureFreshnessScore(feature_name, "static", 0, 1.0, 100.0)

        import re
        for pattern, source_type in self.rolling_patterns:
            match = re.match(pattern, feature_name)
            if match:
                window = int(match.group(1))
                if feature_name in df.columns:
                    missing_rate = df[feature_name].isna().sum() / max(len(df), 1)
                    freshness = max(0, 1 - missing_rate * (window / 5))
                    score = freshness * 100 * max(0, 1 - window / 50)
                    notes = []
                    if missing_rate > 0.2:
                        notes.append(f"High missing rate: {missing_rate:.0%}")
                    return FeatureFreshnessScore(feature_name, source_type, window, freshness, round(score, 1), notes)
        return FeatureFreshnessScore(feature_name, "unknown", 0, 0.5, 50.0, ["Unknown feature type"])

    def analyze_features(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        scores = [self.score_feature(c, df) for c in feature_cols if c in df.columns]
        avg_score = np.mean([s.score for s in scores]) if scores else 0
        warnings = [f"Stale feature: {s.feature_name} (score={s.score:.0f})" for s in scores if s.score < 30]
        return {
            "features_analyzed": len(scores), "average_freshness": round(avg_score, 1),
            "rolling_features": sum(1 for s in scores if s.source_type == "rolling_points"),
            "static_features": sum(1 for s in scores if s.source_type == "static"),
            "stale_features": len(warnings), "warnings": warnings,
        }


# ── Data Quality Report ────────────────────────────────────────────────────


class DataQualityReport:
    """
    Comprehensive data quality assessment combining freshness, leakage, and basic quality checks.
    """

    def __init__(self):
        self.freshness_checker = DataFreshnessChecker()
        self.feature_analyzer = FeatureFreshnessAnalyzer()
        self.leakage_validator = LeakageValidator()

    def generate(self, df: pd.DataFrame, feature_cols: List[str],
                  db_path: Optional[Path] = None, target_col: str = "total_points") -> Dict:
        report = {"timestamp": datetime.now(), "dataset_shape": df.shape}
        if "GAME_DATE" in df.columns:
            dates = pd.to_datetime(df["GAME_DATE"])
            report["date_range"] = {
                "start": str(dates.min().date()), "end": str(dates.max().date()),
                "n_years": round((dates.max() - dates.min()).days / 365.25, 1),
            }
        report["data_quality"] = self._check_data_quality(df, feature_cols)
        report["feature_freshness"] = self.feature_analyzer.analyze_features(df, feature_cols)
        report["leakage"] = self.leakage_validator.check_all(df, feature_cols, target_col)
        report["overall_score"] = self._compute_overall_score(report)
        return report

    def _check_data_quality(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        n_total = len(df)
        missing_counts = df[feature_cols].isna().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        missing_pct = float(missing_counts.sum() / (n_total * len(feature_cols) + 1) * 100)
        dup_cols = [c for c in ["GAME_ID", "GAME_DATE"] if c in df.columns]
        n_dupes = int(df.duplicated(subset=dup_cols).sum()) if dup_cols else 0
        return {
            "missing_values": {"total_missing": int(missing_counts.sum()), "missing_pct": missing_pct,
                               "columns_with_missing": len(cols_with_missing)},
            "duplicate_games": n_dupes,
        }

    def _compute_overall_score(self, report: Dict) -> Dict:
        score = 100.0
        deductions = []
        dq = report.get("data_quality", {})
        missing_pct = dq.get("missing_values", {}).get("missing_pct", 0)
        if missing_pct > 10:
            score -= min(missing_pct, 30)
            deductions.append(f"High missing data ({missing_pct:.0f}%): -{min(missing_pct, 30):.0f}pts")
        leakage = report.get("leakage", {})
        if leakage.get("critical", 0) > 0:
            score -= min(leakage["critical"] * 15, 50)
            deductions.append(f"Critical leakage ({leakage['critical']} issues): -{min(leakage['critical'] * 15, 50):.0f}pts")
        score = max(0, min(100, score))
        grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 50 else "D" if score >= 25 else "F"
        return {"score": round(score, 1), "grade": grade, "deductions": deductions, "passed": score >= 70}
