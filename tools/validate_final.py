"""Final validation: import all modules to verify they compile."""
import sys
import importlib
import traceback

sys.path.insert(0, "src")

modules = [
    ("betting_intel.data.odds_ingestion", "Odds Ingestion"),
    ("betting_intel.betting.ev", "EV Engine"),
    ("betting_intel.betting.bet", "Betting Engine"),
    ("betting_intel.betting.clv", "CLV Tracker"),
    ("betting_intel.features.store", "Feature Store"),
    ("betting_intel.features.builder", "Feature Builder"),
    ("betting_intel.market.movement", "Market Movement"),
    ("betting_intel.market.steam", "Steam Detector"),
    ("betting_intel.market.comparison", "Model vs Market"),
    ("betting_intel.models.ensemble", "Ensemble Model"),
    # predict_tomorrow.py is now a thin entry point at project root,
    # not a module under betting_intel.pipeline. The modular pipeline
    # is accessed via betting_intel.pipeline directly.
    ("betting_intel.pipeline.bootstrap", "Pipeline Bootstrap"),
    ("betting_intel.pipeline.cli", "Pipeline CLI"),
    ("betting_intel.pipeline.data_loading", "Pipeline Data Loading"),
    ("betting_intel.pipeline.modeling", "Pipeline Modeling"),
    ("betting_intel.pipeline.staking", "Pipeline Staking"),
    ("betting_intel.pipeline.risk_analysis", "Pipeline Risk Analysis"),
    ("betting_intel.pipeline.validation", "Pipeline Validation"),
    ("betting_intel.pipeline.reporting", "Pipeline Reporting"),
    ("betting_intel.pipeline.pipeline", "Pipeline Orchestrator"),
    ("betting_intel.pipeline", "Pipeline Package"),
    ("betting_intel.validation.calibration", "Calibration"),
    ("betting_intel.risk.kelly", "Kelly"),
    ("betting_intel.risk.exposure", "Exposure"),
    ("betting_intel.monitoring.drift", "Drift"),
    ("betting_intel.backtesting.metrics", "Backtesting"),
    ("betting_intel.betting.edge", "Edge Analysis"),
]

errors = []
successes = []

for mod_path, name in modules:
    try:
        importlib.import_module(mod_path)
        successes.append(name)
        print(f"  OK  {name}")
    except Exception as e:
        errors.append(f"  FAIL {name}: {e}")
        traceback.print_exc()

print(f"\n{'='*50}")
print(f"  RESULTS: {len(successes)}/{len(modules)} passed, {len(errors)} failed")
print(f"{'='*50}")

if errors:
    print("\nFailures:")
    for e in errors:
        print(f"  {e}")

sys.exit(0 if not errors else 1)
