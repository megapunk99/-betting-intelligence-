"""Validate that all platform modules import correctly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

modules = [
    "betting_intel.betting.ev",
    "betting_intel.betting.clv",
    "betting_intel.betting.bet",
    "betting_intel.data.odds_ingestion",
    "betting_intel.features.store",
    "betting_intel.features.builder",
    "betting_intel.market.movement",
    "betting_intel.market.steam",
    "betting_intel.market.comparison",
    "betting_intel.models.ensemble",
    "betting_intel.pipeline.predict_tomorrow",
]

errors = []
successes = []

for m in modules:
    try:
        exec(f"from {m} import *")
        successes.append(m)
    except Exception as e:
        errors.append(f"{m}: {e}")

print(f"=== IMPORT CHECK ===")
print(f"Success: {len(successes)}/{len(modules)}")
for s in successes:
    print(f"  ✅ {s}")
if errors:
    print(f"\nErrors ({len(errors)}):")
    for e in errors:
        print(f"  ❌ {e}")
