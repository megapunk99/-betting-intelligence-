"""
Configuration — maintained for backward compatibility.
All settings now live in `betting_intel.config`.

New code should import from `betting_intel.config` instead:
    from betting_intel.config import settings, DB_PATH, ...

This file re-exports everything from `betting_intel.config` so existing
scripts (tools, scripts/) continue to work without modification.
"""
import warnings
warnings.warn(
    "Import from 'config' is deprecated. Use 'from betting_intel.config import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from betting_intel.config import *  # noqa: F401, F403
