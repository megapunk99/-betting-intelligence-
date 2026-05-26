"""
Main pipeline orchestrator — wraps the root-level BettingIntelligenceSystem
for use through the CLI and API.
"""
import sys
import os
from pathlib import Path

# Add project root to path so root-level imports work
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Re-export from root-level main module
from main import BettingIntelligenceSystem, main as run_pipeline

__all__ = ["BettingIntelligenceSystem", "run_pipeline"]
