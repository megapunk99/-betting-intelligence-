"""Fix remaining lint errors that ruff could not auto-fix."""

import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)


def fix_file(path, fixes):
    """Apply a list of fixes to a file."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new, 1)
        else:
            print(f"  [SKIP] {path}: could not find:\n    {repr(old[:80])}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  [OK] {path}")


def main():
    # ── F821: Undefined names ──────────────────────────────────────────

    # src/betting_intel/cli/main.py: add import os
    fix_file("src/betting_intel/cli/main.py", [
        ("import json\nfrom pathlib import Path\nfrom datetime import datetime\n\nimport click\n\nfrom betting_intel.config import settings\nimport logging",
         "import json\nimport os\nfrom pathlib import Path\nfrom datetime import datetime\n\nimport click\n\nfrom betting_intel.config import settings\nimport logging"),
    ])

    # src/betting_intel/live/odds_fetcher.py: F821 'self' in staticmethod
    # The self._check_quota_warnings issue - need to check if it's a static method issue
    # Looking at the output, it was in the class - need to check the exact issue
    print("  [MANUAL CHECK NEEDED] src/betting_intel/live/odds_fetcher.py: self in staticmethod")

    # ── F601: Duplicate dictionary key ──────────────────────────────────
    # tests/test_analytics_tracker.py: "bet_type" key repeated
    with open("tests/test_analytics_tracker.py", 'r', encoding='utf-8') as f:
        content = f.read()
    # Fix: replace the second "bet_type" with a unique key
    old = '''                            "bet_type": "total",
                        },
                    ],
                    "source": {\"engine\": 1},
                },
            ),'''
    new = '''                            "bet_type": "total",
                        },
                    ],
                    "sources": {\"engine\": 1},
                },
            ),'''
    if old in content:
        content = content.replace(old, new, 1)
        with open("tests/test_analytics_tracker.py", 'w', encoding='utf-8') as f:
            f.write(content)
        print("  [OK] tests/test_analytics_tracker.py: fixed duplicate key")
    else:
        print("  [SKIP] tests/test_analytics_tracker.py: pattern not found")

    # ── F841: Unused variables ─────────────────────────────────────────
    # These need to be removed one by one. Let me do the simpler ones.

    # src/betting_intel/data/draftkings_scraper.py: remove unused now_utc
    with open("src/betting_intel/data/draftkings_scraper.py", 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace("now_utc = datetime.now(timezone.utc)\n\n        for event in events:\n            try:\n                game = cls._parse_dk_event(event)\n                if game:\n                    parsed_games.append(game)\n            except Exception as e:\n                logger.debug(f\"Skipping malformed DK event: {e}\")\n                continue\n\n        if parsed_games:\n            logger.info(f\"DraftKings parsed: {len(parsed_games)} games with odds\")\n\n        return parsed_games\n\n    @classmethod\n    def _parse_dk_event(cls, event: dict) -> Optional[dict]:", 
        "parsed_games: list[dict] = []\n\n        for event in events:\n            try:\n                game = cls._parse_dk_event(event)\n                if game:\n                    parsed_games.append(game)\n            except Exception as e:\n                logger.debug(f\"Skipping malformed DK event: {e}\")\n                continue\n\n        if parsed_games:\n            logger.info(f\"DraftKings parsed: {len(parsed_games)} games with odds\")\n\n        return parsed_games\n\n    @classmethod\n    def _parse_dk_event(cls, event: dict) -> Optional[dict]:")
    with open("src/betting_intel/data/draftkings_scraper.py", 'w', encoding='utf-8') as f:
        f.write(content)
    print("  [OK] src/betting_intel/data/draftkings_scraper.py: removed unused now_utc")

    # src/betting_intel/live/engine.py: remove unused is_seeded
    with open("src/betting_intel/live/engine.py", 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace("all_games: list[LiveGame] = []\n        is_seeded = False\n\n        if raw_odds:", 
        "all_games: list[LiveGame] = []\n\n        if raw_odds:")
    with open("src/betting_intel/live/engine.py", 'w', encoding='utf-8') as f:
        f.write(content)
    print("  [OK] src/betting_intel/live/engine.py: removed unused is_seeded")

    # src/betting_intel/data/odds_fetcher.py: remove unused 'overs', 'unders', 'titles', 'last_home', 'last_away'
    with open("src/betting_intel/data/odds_fetcher.py", 'r', encoding='utf-8') as f:
        content = f.read()
    # Fix: remove unused vars in _compute_totals_consensus
    old1 = '''def _compute_totals_consensus(books: List[BookOdds]) -> Dict:\n    \"\"\"Aggregate totals across all books.\"\"\"\n    overs = [b.total_over for b in books]\n    unders = [b.total_under for b in books]\n    over_odds = [b.total_over_odds for b in books]\n    under_odds = [b.total_under_odds for b in books]\n    titles = [b.book_title for b in books]'''
    new1 = '''def _compute_totals_consensus(books: List[BookOdds]) -> Dict:\n    \"\"\"Aggregate totals across all books.\"\"\"\n    over_odds = [b.total_over_odds for b in books]\n    under_odds = [b.total_under_odds for b in books]'''
    content = content.replace(old1, new1, 1)
    # Fix: remove unused 'name' in stealth_scraper
    with open("src/betting_intel/data/stealth_scraper.py", 'r', encoding='utf-8') as f:
        sc = f.read()
    sc = sc.replace("name = o.get(\"displayName\", \"\")\n            is_home = c.get(\"homeAway\") == \"home\"",
                    "is_home = c.get(\"homeAway\") == \"home\"")
    with open("src/betting_intel/data/stealth_scraper.py", 'w', encoding='utf-8') as f:
        f.write(sc)
    print("  [OK] src/betting_intel/data/stealth_scraper.py: removed unused 'name'")
    with open("src/betting_intel/data/odds_fetcher.py", 'w', encoding='utf-8') as f:
        f.write(content)
    print("  [OK] src/betting_intel/data/odds_fetcher.py: removed unused vars")

    # ── F401: Unused imports ──────────────────────────────────────────
    # src/betting_intel/models/hyperparameter_tuning.py: remove unused lightgbm early_stopping imports
    with open("src/betting_intel/models/hyperparameter_tuning.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix: remove the unused imports at the bottom
    old_lgb = """# Compatibility import for LGBM early stopping\ntry:\n    from lightgbm import early_stopping as LGBMEarlyStopping\nexcept ImportError:\n    try:\n        from lightgbm.callback import early_stopping as LGBMEarlyStopping\n    except ImportError:\n        LGBMEarlyStopping = None"""
    new_lgb = """# LGBMEarlyStopping is imported inside the LightGBM tuning method\n# where it's used (lazy import avoids import errors on older versions)\nLGBMEarlyStopping = None  # Fallback: set inside tune_lightgbm"""
    content = content.replace(old_lgb, new_lgb, 1)
    with open("src/betting_intel/models/hyperparameter_tuning.py", 'w', encoding='utf-8') as f:
        f.write(content)
    print("  [OK] src/betting_intel/models/hyperparameter_tuning.py: fixed LGBM imports")

    # ── E702: Multiple statements on one line ──────────────────────────
    # tests/test_live_engine.py: semicolons
    with open("tests/test_live_engine.py", 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace(";", "\n                    ")
    with open("tests/test_live_engine.py", 'w', encoding='utf-8') as f:
        f.write(content)
    print("  [OK] tests/test_live_engine.py: removed semicolons")

    print("\nDone. Run 'ruff check src/ tests/' to see remaining errors.")


if __name__ == "__main__":
    main()
