"""Validate that all platform modules import correctly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Original modules
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
    # predict_tomorrow.py is now a thin entry point at project root,
    # not a module under betting_intel.pipeline. The modular pipeline
    # is accessed via betting_intel.pipeline directly.
]

# New modular pipeline components
pipeline_modules = [
    "betting_intel.pipeline.bootstrap",
    "betting_intel.pipeline.cli",
    "betting_intel.pipeline.data_loading",
    "betting_intel.pipeline.modeling",
    "betting_intel.pipeline.staking",
    "betting_intel.pipeline.risk_analysis",
    "betting_intel.pipeline.validation",
    "betting_intel.pipeline.reporting",
    "betting_intel.pipeline.pipeline",
    "betting_intel.pipeline",
]

errors = []
successes = []

for m in modules + pipeline_modules:
    try:
        exec(f"from {m} import *")
        successes.append(m)
    except Exception as e:
        errors.append(f"{m}: {e}")

print(f"=== IMPORT CHECK ===")
print(f"Success: {len(successes)}/{len(modules + pipeline_modules)}")
for s in successes:
    print(f"  ✅ {s}")
if errors:
    print(f"\nErrors ({len(errors)}):")
    for e in errors:
        print(f"  ❌ {e}")
