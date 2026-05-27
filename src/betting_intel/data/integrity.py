"""
Data integrity module: staleness detection, feature freshness scoring,
and data leakage validation.

Key features:
1. Stale data detection — flag when data sources haven't been updated
2. Feature freshness scoring — score how recent each feature's source data is
3. Leakage validation — automatically check for lookahead bias in feature engineering
4. Data quality scoring — composite score of dataset health
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

    Detects:
    - Databases that haven't been updated recently
    - Cache files that are stale
    - Missing data for recent dates
    """

    def __init__(self, staleness_thresholds: Optional[Dict[str, float]] = None):
        """
        Args:
            staleness_thresholds: Dict of {source_name: max_age_hours}
                Default: DB=24h, cache=1h, odds=0.25h (15 min)
        """
        self.staleness_thresholds = staleness_thresholds or {
            "nba_db": 24.0,       # NBA data: 24 hours
            "cache": 1.0,         # Cache files: 1 hour
            "odds_api": 0.25,     # OddsAPI: 15 minutes
            "model": 168.0,       # Saved models: 7 days
            "output": 24.0,       # Output files: 24 hours
        }

    def check_file(self, path: Path, source_type: str = "nba_db") -> DataSourceStatus:
        """Check freshness of a single file."""
        threshold = self.staleness_thresholds.get(source_type, 24.0)

        if not path.exists():
            return DataSourceStatus(
                name=path.name,
                path=path,
                last_modified=None,
                age_hours=float("inf"),
                staleness_threshold_hours=threshold,
                is_fresh=False,
                n_records=0,
                notes=["FILE_NOT_FOUND"],
            )

        # Get file stats
        stat = path.stat()
        last_modified = datetime.fromtimestamp(stat.st_mtime)
        age_hours = (datetime.now() - last_modified).total_seconds() / 3600

        is_fresh = age_hours <= threshold

        notes = []
        if not is_fresh:
            overdue_pct = (age_hours - threshold) / threshold * 100
            notes.append(f"STALE: {age_hours:.1f}h old (threshold: {threshold}h, {overdue_pct:.0f}% overdue)")

        return DataSourceStatus(
            name=path.name,
            path=path,
            last_modified=last_modified,
            age_hours=age_hours,
            staleness_threshold_hours=threshold,
            is_fresh=is_fresh,
            notes=notes,
        )

    def check_database(
        self,
        db_path: Path,
        table_names: List[str],
        date_column: str = "GAME_DATE",
    ) -> DataSourceStatus:
        """Check database freshness by looking at most recent date in tables."""
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
                        cursor = conn.execute(
                            f"SELECT MAX({date_column}) FROM {table}"
                        )
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
                name=db_path.name,
                path=db_path,
                last_modified=None,
                age_hours=float("inf"),
                staleness_threshold_hours=threshold,
                is_fresh=False,
                n_records=0,
                notes=[f"DB_ERROR: {str(e)}"],
            )

        if latest_date is None:
            return DataSourceStatus(
                name=db_path.name,
                path=db_path,
                last_modified=None,
                age_hours=float("inf"),
                staleness_threshold_hours=threshold,
                is_fresh=False,
                n_records=total_records,
                notes=["NO_DATA: No records found in any table"],
            )

        days_since_update = (datetime.now().date() - latest_date).days
        is_fresh = days_since_update <= threshold / 24

        notes = []
        if not is_fresh:
            notes.append(
                f"STALE: Most recent data is {days_since_update} days old"
            )

        return DataSourceStatus(
            name=db_path.name,
            path=db_path,
            last_modified=datetime.combine(latest_date, datetime.min.time()),
            age_hours=days_since_update * 24,
            staleness_threshold_hours=threshold,
            is_fresh=is_fresh,
            n_records=total_records,
            date_range=(date(2020, 1, 1), latest_date) if latest_date else None,
            notes=notes,
        )

    def get_freshness_score(self, statuses: List[DataSourceStatus]) -> Dict:
        """Compute overall data freshness score (0-100)."""
        if not statuses:
            return {"score": 0, "grade": "F", "details": "No data sources"}

        fresh_count = sum(1 for s in statuses if s.is_fresh)
        total_count = len(statuses)

        freshness_ratio = fresh_count / max(total_count, 1)

        # Age penalties
        age_penalties = []
        for s in statuses:
            if s.age_hours > s.staleness_threshold_hours * 3:
                age_penalties.append(1.0)  # Fully penalized
            elif s.age_hours > s.staleness_threshold_hours:
                age_penalties.append(
                    (s.age_hours - s.staleness_threshold_hours)
                    / (s.staleness_threshold_hours * 2)
                )
            else:
                age_penalties.append(0.0)

        avg_penalty = np.mean(age_penalties) if age_penalties else 0
        score = max(0, freshness_ratio * 100 - avg_penalty * 50)

        if score >= 90:
            grade = "A"
        elif score >= 75:
            grade = "B"
        elif score >= 50:
            grade = "C"
        elif score >= 25:
            grade = "D"
        else:
            grade = "F"

        return {
            "score": round(score, 1),
            "grade": grade,
            "fresh_sources": fresh_count,
            "total_sources": total_count,
            "freshness_ratio": round(freshness_ratio, 2),
            "statuses": statuses,
        }


# ── Feature Freshness Scoring ──────────────────────────────────────────────


@dataclass
class FeatureFreshnessScore:
    """Score for how fresh a feature is."""

    feature_name: str
    source_type: str  # 'rolling', 'seasonal', 'static'
    source_window: int  # Number of games in the rolling window
    recency: float  # 0 (stale) to 1 (fresh)
    score: float  # 0-100 composite freshness score
    notes: List[str] = field(default_factory=list)


class FeatureFreshnessAnalyzer:
    """
    Analyzes freshness of engineered features.

    Different feature types have different freshness expectations:
    - Rolling features (10-game avg): needs recent data to be fresh
    - Seasonal features (avg by season): fresh until new season
    - Static features (team_id, home/away): always fresh
    """

    def __init__(self):
        self.rolling_patterns = [
            (r"avg_pts_(\d+)g", "rolling_points"),
            (r"avg_pace_(\d+)g", "rolling_pace"),
            (r"avg_efg_(\d+)g", "rolling_efg"),
            (r"avg_pm_(\d+)g", "rolling_plus_minus"),
            (r"avg_ts_(\d+)g", "rolling_ts"),
        ]

    def score_feature(self, feature_name: str, df: pd.DataFrame) -> FeatureFreshnessScore:
        """Score the freshness of a single feature."""
        # Static features (always fresh)
        static_patterns = [
            "IS_HOME", "home_", "team_", "GAME_", "SEASON_",
            "TEAM_", "rest_", "is_", "has_",
        ]
        if any(feature_name.startswith(p) for p in static_patterns) and \
           not any(kw in feature_name for kw in ["rolling", "avg_", "cum_"]):
            return FeatureFreshnessScore(
                feature_name=feature_name,
                source_type="static",
                source_window=0,
                recency=1.0,
                score=100.0,
            )

        # Rolling features — check window size and missing count
        for pattern, source_type in self.rolling_patterns:
            import re
            match = re.match(pattern, feature_name)
            if match:
                window = int(match.group(1))
                if feature_name in df.columns:
                    n_missing = df[feature_name].isna().sum()
                    n_total = len(df)
                    missing_rate = n_missing / max(n_total, 1)

                    # More penalizing for larger windows with more missing
                    freshness = max(0, 1 - missing_rate * (window / 5))
                    score = freshness * 100 * max(0, 1 - window / 50)

                    notes = []
                    if missing_rate > 0.2:
                        notes.append(f"High missing rate: {missing_rate:.0%}")
                    if window > 20:
                        notes.append(f"Large window ({window}g) — slower to adapt")

                    return FeatureFreshnessScore(
                        feature_name=feature_name,
                        source_type=source_type,
                        source_window=window,
                        recency=freshness,
                        score=round(score, 1),
                        notes=notes,
                    )

        # Default: moderate freshness
        return FeatureFreshnessScore(
            feature_name=feature_name,
            source_type="unknown",
            source_window=0,
            recency=0.5,
            score=50.0,
            notes=["Unknown feature type"],
        )

    def analyze_features(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Score freshness of all features and return summary."""
        scores = []
        for col in feature_cols:
            if col in df.columns:
                scores.append(self.score_feature(col, df))

        avg_score = np.mean([s.score for s in scores]) if scores else 0
        rolling_scores = [s for s in scores if s.source_type == "rolling_points"]
        static_count = sum(1 for s in scores if s.source_type == "static")

        warnings = []
        for s in scores:
            if s.score < 30:
                warnings.append(f"Stale feature: {s.feature_name} (score={s.score:.0f})")

        return {
            "features_analyzed": len(scores),
            "average_freshness": round(avg_score, 1),
            "rolling_features": len(rolling_scores),
            "static_features": static_count,
            "stale_features": len(warnings),
            "warnings": warnings,
            "feature_scores": {s.feature_name: s.score for s in scores},
        }


# ── Data Leakage Validation ────────────────────────────────────────────────


class LeakageValidator:
    """
    Validates that feature engineering does not introduce lookahead bias.

    Checks:
    1. Any feature using shift(0) instead of shift(1) (current game leaks)
    2. Any rolling window without shift (includes current game)
    3. Train/test temporal ordering (no future data in training)
    4. Market line in features (benchmark leakage)
    5. Target encoding without grouping

    Usage:
        validator = LeakageValidator()
        issues = validator.check_feature_engineering(feature_df, target_col="total_points")
        print(validator.format_report(issues))
    """

    def __init__(self):
        self.checks_run: List[str] = []
        self.issues: List[Dict] = []

    def check_rolling_window_leakage(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        group_cols: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Check that rolling features use shift(1) (exclude current game).

        Detects columns that look like rolling features but aren't shifted.
        """
        issues = []

        for col in feature_cols:
            if col not in df.columns:
                continue

            # Check if this looks like a rolling feature
            is_rolling = any(
                kw in col.lower() for kw in [
                    "rolling", "_avg", "avg_", "_sum", "cum_",
                    "_mean", "_std", "trailing",
                ]
            )

            if not is_rolling:
                continue

            # Check correlation with target — if too high, might be leaked
            target_col = "total_points"
            if target_col in df.columns:
                try:
                    # Sample to avoid slow computation on huge datasets
                    sample = df[[col, target_col]].dropna().sample(min(1000, len(df)), random_state=42)
                    corr = sample[col].corr(sample[target_col])

                    # Rolling window median vs actual points should not be > 0.95
                    # (that would mean it's basically the target)
                    if abs(corr) > 0.95:
                        issues.append({
                            "severity": "critical",
                            "feature": col,
                            "issue": "SUSPICIOUSLY HIGH CORRELATION WITH TARGET",
                            "detail": f"Correlation with {target_col}: {corr:.3f}. "
                                      f"May be leaking target information.",
                            "recommendation": f"Check if {col} uses shift(0) instead of shift(1).",
                        })
                except Exception:
                    continue

        return issues

    def check_market_line_leakage(
        self,
        feature_cols: List[str],
        market_cols: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Check that market line columns are not in features."""
        if market_cols is None:
            market_cols = [
                "market_line_baseline",
                "market_line_pace_adj",
                "trailing_avg_total_10g",
            ]

        issues = []
        for mc in market_cols:
            if mc in feature_cols:
                issues.append({
                    "severity": "critical",
                    "feature": mc,
                    "issue": "MARKET LINE IN FEATURES",
                    "detail": f"'{mc}' should not be used as a model feature. "
                              f"It represents the benchmark to beat, not an input.",
                    "recommendation": f"Exclude '{mc}' from feature selection. "
                                      f"Use only in backtesting for edge calculation.",
                })

        return issues

    def check_temporal_ordering(
        self,
        df: pd.DataFrame,
        date_col: str = "GAME_DATE",
    ) -> List[Dict]:
        """
        Check for temporal ordering issues.

        Looks for:
        - Duplicate dates across teams (means both teams' data is available)
        - Games out of order
        - Missing date column
        """
        issues = []

        if date_col not in df.columns:
            issues.append({
                "severity": "critical",
                "feature": "GAME_DATE",
                "issue": "MISSING DATE COLUMN",
                "detail": "No date column found. Cannot validate temporal ordering.",
                "recommendation": "Add a date column to enable time-series validation.",
            })
            return issues

        dates = pd.to_datetime(df[date_col])
        if not dates.is_monotonic_increasing:
            issues.append({
                "severity": "warning",
                "feature": date_col,
                "issue": "DATA NOT SORTED CHRONOLOGICALLY",
                "detail": f"Data should be sorted by {date_col} for proper time-series processing.",
                "recommendation": "Sort by date before feature engineering and backtesting.",
            })

        return issues

    def check_target_leakage(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "total_points",
        threshold: float = 0.98,
    ) -> List[Dict]:
        """
        Check for target leakage using correlation analysis.

        Any feature with near-perfect correlation to target is likely leaked.
        """
        issues = []
        target = df[target_col]

        for col in feature_cols:
            if col == target_col or col not in df.columns:
                continue

            try:
                sample = df[[col]].dropna().sample(min(500, len(df)), random_state=42)
                if len(sample) < 10:
                    continue

                y_sample = target.loc[sample.index]
                corr = float(sample[col].corr(y_sample))

                if abs(corr) >= threshold:
                    issues.append({
                        "severity": "critical",
                        "feature": col,
                        "issue": "POTENTIAL TARGET LEAKAGE",
                        "detail": f"Correlation with target = {corr:.4f} (threshold: {threshold})",
                        "recommendation": f"Investigate '{col}'. It may include future information. "
                                          f"Check for shift(0) instead of shift(1) or direct use of target.",
                    })
            except Exception:
                continue

        return issues

    def check_all(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "total_points",
        date_col: str = "GAME_DATE",
    ) -> Dict:
        """Run all leakage checks and return comprehensive report."""
        self.issues = []

        # 1. Rolling window leakage
        self.issues.extend(self.check_rolling_window_leakage(df, feature_cols))

        # 2. Market line leakage
        self.issues.extend(self.check_market_line_leakage(feature_cols))

        # 3. Temporal ordering
        self.issues.extend(self.check_temporal_ordering(df, date_col))

        # 4. Target leakage
        self.issues.extend(self.check_target_leakage(df, feature_cols, target_col))

        # Severity counts
        n_critical = sum(1 for i in self.issues if i["severity"] == "critical")
        n_warnings = sum(1 for i in self.issues if i["severity"] == "warning")

        return {
            "n_checks": len(self.checks_run),
            "n_issues": len(self.issues),
            "critical": n_critical,
            "warnings": n_warnings,
            "passed": n_critical == 0,
            "issues": self.issues,
            "recommendation": (
                "No leakage detected. Feature engineering is clean."
                if n_critical == 0
                else f"Fix {n_critical} critical issue(s) before using model predictions."
            ),
        }

    def format_report(self, results: Dict) -> str:
        """Format leakage validation results as readable text."""
        lines = [
            "=" * 60,
            "  DATA LEAKAGE VALIDATION REPORT",
            "=" * 60,
        ]

        if results["passed"]:
            lines.extend([
                "",
                "  STATUS: PASSED — No critical leakage detected",
                f"  ({results['n_issues']} total issues: "
                f"{results['critical']} critical, {results['warnings']} warnings)",
                "",
            ])
        else:
            lines.extend([
                "",
                f"  STATUS: FAILED — {results['critical']} critical issue(s) found",
                f"  ({results['n_issues']} total issues)",
                "",
                "  -- Critical Issues --",
            ])
            for issue in results["issues"]:
                if issue["severity"] == "critical":
                    lines.extend([
                        f"  [{issue['severity'].upper()}] {issue['feature']}",
                        f"  Issue: {issue['issue']}",
                        f"  Detail: {issue['detail']}",
                        f"  Fix: {issue['recommendation']}",
                        "",
                    ])

            if results["warnings"] > 0:
                lines.extend(["  -- Warnings --"])
                for issue in results["issues"]:
                    if issue["severity"] == "warning":
                        lines.extend([
                            f"  [WARN] {issue['feature']}: {issue['issue']}",
                            f"         {issue['recommendation']}",
                        ])

        lines.append("=" * 60)
        return "\n".join(lines)


# ── Overall Data Quality Score ─────────────────────────────────────────────


class DataQualityReport:
    """
    Comprehensive data quality assessment.

    Combines:
    - Freshness score (from DataFreshnessChecker)
    - Feature freshness (from FeatureFreshnessAnalyzer)
    - Leakage validation (from LeakageValidator)
    - Basic data quality checks (missing values, outliers, completeness)

    Usage:
        reporter = DataQualityReport()
        report = reporter.generate(
            df=feature_df,
            feature_cols=feature_cols,
            db_path=DB_PATH,
        )
        print(reporter.format(report))
    """

    def __init__(self):
        self.freshness_checker = DataFreshnessChecker()
        self.feature_analyzer = FeatureFreshnessAnalyzer()
        self.leakage_validator = LeakageValidator()

    def generate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        db_path: Optional[Path] = None,
        target_col: str = "total_points",
    ) -> Dict:
        """Generate comprehensive data quality report."""
        report = {
            "timestamp": datetime.now(),
            "dataset_shape": df.shape,
            "date_range": None,
        }

        # Date range
        if "GAME_DATE" in df.columns:
            dates = pd.to_datetime(df["GAME_DATE"])
            report["date_range"] = {
                "start": str(dates.min().date()),
                "end": str(dates.max().date()),
                "n_years": round((dates.max() - dates.min()).days / 365.25, 1),
            }

        # 1. Basic data quality
        report["data_quality"] = self._check_data_quality(df, feature_cols)

        # 2. Feature freshness
        report["feature_freshness"] = self.feature_analyzer.analyze_features(df, feature_cols)

        # 3. Leakage validation
        report["leakage"] = self.leakage_validator.check_all(df, feature_cols, target_col)

        # 4. Database freshness
        if db_path and db_path.exists():
            import sqlite3
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = [row[0] for row in cursor.fetchall()]
                conn.close()
                report["db_freshness"] = self.freshness_checker.check_database(
                    db_path, tables
                )
            except Exception:
                report["db_freshness"] = None
        else:
            report["db_freshness"] = None

        # 5. Overall score
        report["overall_score"] = self._compute_overall_score(report)

        return report

    def _check_data_quality(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Basic data quality checks."""
        n_total = len(df)
        quality = {}

        # Missing values
        missing_counts = df[feature_cols].isna().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        quality["missing_values"] = {
            "total_missing": int(missing_counts.sum()),
            "missing_pct": float(missing_counts.sum() / (n_total * len(feature_cols) + 1) * 100),
            "columns_with_missing": len(cols_with_missing),
            "worst_features": [
                {"col": col, "missing_pct": float(count / n_total * 100)}
                for col, count in cols_with_missing.nlargest(5).items()
            ],
        }

        # Infinite values
        n_infinite = sum(
            np.isinf(df[col]).sum() for col in feature_cols if col in df.columns and df[col].dtype.kind in "fc"
        )
        quality["infinite_values"] = int(n_infinite)

        # Duplicate rows
        dup_cols = [c for c in ["GAME_ID", "GAME_DATE"] if c in df.columns]
        if dup_cols:
            n_duplicates = df.duplicated(subset=dup_cols).sum()
            quality["duplicate_games"] = int(n_duplicates)
        else:
            quality["duplicate_games"] = 0

        # Outlier detection (features with extreme z-scores)
        outlier_features = []
        for col in feature_cols:
            if col in df.columns and df[col].dtype.kind in "fc":
                z_scores = np.abs((df[col] - df[col].mean()) / max(df[col].std(), 1e-6))
                n_outliers = (z_scores > 4).sum()
                if n_outliers > max(5, n_total * 0.01):
                    outlier_features.append({
                        "col": col,
                        "n_outliers": int(n_outliers),
                        "outlier_pct": float(n_outliers / n_total * 100),
                    })
        quality["outliers"] = {
            "features_with_outliers": len(outlier_features),
            "top_features": sorted(outlier_features, key=lambda x: -x["outlier_pct"])[:5],
        }

        return quality

    def _compute_overall_score(self, report: Dict) -> Dict:
        """Compute overall data quality score (0-100)."""
        score = 100.0
        deductions = []
        details = []

        # Data quality deduction
        dq = report.get("data_quality", {})
        if dq:
            missing_pct = dq.get("missing_values", {}).get("missing_pct", 0)
            if missing_pct > 10:
                deduction = min(missing_pct, 30)
                score -= deduction
                deductions.append(f"High missing data ({missing_pct:.0f}%): -{deduction:.0f}pts")

        # Leakage deduction
        leakage = report.get("leakage", {})
        if leakage:
            n_critical = leakage.get("critical", 0)
            if n_critical > 0:
                deduction = min(n_critical * 15, 50)
                score -= deduction
                deductions.append(f"Critical leakage ({n_critical} issues): -{deduction:.0f}pts")

        # Freshness deduction
        ff = report.get("feature_freshness", {})
        if ff:
            avg_freshness = ff.get("average_freshness", 100)
            if avg_freshness < 50:
                deduction = min((50 - avg_freshness) * 0.5, 20)
                score -= deduction
                deductions.append(f"Low feature freshness ({avg_freshness:.0f}/100): -{deduction:.0f}pts")

        # DB freshness deduction
        db = report.get("db_freshness")
        if db and not db.is_fresh:
            deduction = 10
            score -= deduction
            deductions.append(f"Stale database ({db.age_hours:.0f}h old): -{deduction:.0f}pts")

        score = max(0, min(100, score))

        grade = (
            "A" if score >= 90
            else "B" if score >= 75
            else "C" if score >= 50
            else "D" if score >= 25
            else "F"
        )

        return {
            "score": round(score, 1),
            "grade": grade,
            "deductions": deductions,
            "passed": score >= 70,
        }

    def format(self, report: Dict) -> str:
        """Format comprehensive data quality report."""
        lines = [
            "=" * 60,
            "  DATA QUALITY REPORT",
            "=" * 60,
            f"  Generated: {report['timestamp'].strftime('%Y-%m-%d %H:%M')}",
            "",
            f"  Dataset: {report['dataset_shape'][0]:,} rows x {report['dataset_shape'][1]:,} cols",
        ]

        if report.get("date_range"):
            dr = report["date_range"]
            lines.append(
                f"  Date range: {dr['start']} to {dr['end']} ({dr['n_years']} years)"
            )

        # Overall score
        overall = report.get("overall_score", {})
        grade = overall.get("grade", "N/A")
        score = overall.get("score", 0)
        lines.extend([
            "",
            f"  OVERALL QUALITY: {grade} ({score:.0f}/100)",
        ])
        for d in overall.get("deductions", []):
            lines.append(f"    - {d}")

        # Data quality details
        dq = report.get("data_quality", {})
        if dq:
            lines.extend([
                "",
                "  -- Data Quality --",
                f"  Missing values: {dq.get('missing_values', {}).get('total_missing', 0):,} "
                f"({dq.get('missing_values', {}).get('missing_pct', 0):.1f}%)",
                f"  Features with missing: {dq.get('missing_values', {}).get('columns_with_missing', 0)}",
                f"  Duplicate games: {dq.get('duplicate_games', 0)}",
            ])

        # Feature freshness
        ff = report.get("feature_freshness", {})
        if ff:
            lines.extend([
                "",
                "  -- Feature Freshness --",
                f"  Average score: {ff.get('average_freshness', 0):.1f}/100",
                f"  Rolling features: {ff.get('rolling_features', 0)}",
                f"  Stale features: {ff.get('stale_features', 0)}",
            ])
            for w in ff.get("warnings", [])[:5]:
                lines.append(f"    [!] {w}")

        # Leakage
        leakage = report.get("leakage", {})
        if leakage:
            lines.extend([
                "",
                "  -- Leakage Check --",
                f"  Status: {'PASSED' if leakage.get('passed') else 'FAILED'}",
                f"  Issues: {leakage.get('critical', 0)} critical, "
                f"{leakage.get('warnings', 0)} warnings",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)
