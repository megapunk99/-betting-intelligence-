"""
Safe serialization utilities — protects against unsafe deserialization.

All model files (joblib, pickle) are accompanied by a SHA-256 .hash file.
Before loading, the hash is verified to ensure the file hasn't been
tampered with or corrupted.

Architecture:
    model.joblib  +  model.joblib.hash  (SHA-256 hex digest)
    data.pkl      +  data.pkl.hash      (SHA-256 hex digest)

Usage:
    from betting_intel.utils.safe_serialize import (
        safe_joblib_load, safe_joblib_dump,
        safe_pickle_load, safe_pickle_dump,
        compute_file_hash, verify_file_hash,
        ModelIntegrityError,
    )

    # Save with hash
    safe_joblib_dump(my_model, "models/total_model.joblib")

    # Load with hash verification (auto-verifies against .hash file)
    model = safe_joblib_load("models/total_model.joblib")

    # Load with explicit hash (override auto-verification)
    model = safe_joblib_load("models/total_model.joblib", expected_hash="abc123...")
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# ── Error Types ────────────────────────────────────────────────────────────


class ModelIntegrityError(Exception):
    """Raised when a model file fails hash verification."""


class ModelNotFoundError(FileNotFoundError):
    """Raised when a model file is not found."""


# ── Hashing ────────────────────────────────────────────────────────────────


def compute_file_hash(path: Union[str, Path]) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        path: Path to the file.

    Returns:
        SHA-256 hex digest string.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file_hash(path: Union[str, Path], expected_hash: str) -> bool:
    """Verify that a file matches an expected SHA-256 hash.

    Args:
        path: Path to the file.
        expected_hash: Expected SHA-256 hex digest.

    Returns:
        True if hash matches.

    Raises:
        ModelIntegrityError: If the hash does NOT match.
        FileNotFoundError: If the file doesn't exist.
    """
    actual = compute_file_hash(path)
    if actual != expected_hash:
        raise ModelIntegrityError(
            f"Hash mismatch for {path}: "
            f"expected {expected_hash[:16]}..., "
            f"got {actual[:16]}..."
        )
    return True


# ── Hash File Management ──────────────────────────────────────────────────


def _hash_path(path: Path) -> Path:
    """Return the companion .hash file path."""
    return path.with_name(path.name + ".hash")


def _read_hash_file(path: Path) -> Optional[str]:
    """Read the expected hash from a .hash file.

    Returns:
        Hash string, or None if the .hash file doesn't exist.
    """
    hpath = _hash_path(path)
    if not hpath.exists():
        return None
    try:
        return hpath.read_text().strip()
    except Exception:
        logger.warning(f"Failed to read hash file: {hpath}")
        return None


def _write_hash_file(path: Path, hash_str: str) -> None:
    """Write a hash to the companion .hash file."""
    hpath = _hash_path(path)
    try:
        hpath.write_text(hash_str + "\n")
    except Exception as e:
        logger.warning(f"Failed to write hash file {hpath}: {e}")


# ── Joblib Serialization ──────────────────────────────────────────────────


def safe_joblib_dump(obj: Any, path: Union[str, Path]) -> str:
    """Save an object with joblib.dump() and compute a SHA-256 hash.

    The hash is written to a companion .hash file for later verification.

    Args:
        obj: Object to serialize.
        path: File path to save to.

    Returns:
        SHA-256 hex digest of the saved file.

    Raises:
        IOError: If writing fails.
    """
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(obj, path)
    file_hash = compute_file_hash(path)
    _write_hash_file(path, file_hash)

    logger.debug(f"Saved {path} with hash {file_hash[:16]}...")
    return file_hash


def safe_joblib_load(
    path: Union[str, Path],
    expected_hash: Optional[str] = None,
    verify: bool = True,
) -> Any:
    """Load an object with joblib.load() after verifying hash integrity.

    Args:
        path: File path to load from.
        expected_hash: If provided, verify against this hash.
                       If None and a .hash file exists, use that.
                       If None and no .hash file exists and verify=True,
                       load is rejected with ModelIntegrityError.
        verify: If True (default), require hash verification.
                If False, load without verification (warns).

    Returns:
        Deserialized object.

    Raises:
        ModelIntegrityError: If hash verification fails.
        FileNotFoundError: If the file doesn't exist.
    """
    import joblib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if verify:
        # Resolve the expected hash
        resolved_hash = expected_hash
        if resolved_hash is None:
            resolved_hash = _read_hash_file(path)

        if resolved_hash is None:
            raise ModelIntegrityError(
                f"No hash available for {path}. "
                f"No .hash file found and no expected_hash provided. "
                f"Run safe_joblib_dump() to create one, or set verify=False "
                f"to bypass (not recommended for production)."
            )

        verify_file_hash(path, resolved_hash)
        logger.debug(f"Hash verified for {path}")
    else:
        logger.warning(f"Loading {path} WITHOUT hash verification — unsafe!")

    obj = joblib.load(path)
    return obj


# ── Pickle Serialization ──────────────────────────────────────────────────


def safe_pickle_dump(obj: Any, path: Union[str, Path]) -> str:
    """Save an object with pickle.dump() and compute a SHA-256 hash.

    Args:
        obj: Object to serialize.
        path: File path to save to.

    Returns:
        SHA-256 hex digest of the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_hash = compute_file_hash(path)
    _write_hash_file(path, file_hash)

    logger.debug(f"Saved {path} with hash {file_hash[:16]}...")
    return file_hash


def safe_pickle_load(
    path: Union[str, Path],
    expected_hash: Optional[str] = None,
    verify: bool = True,
) -> Any:
    """Load an object with pickle.load() after verifying hash integrity.

    Args:
        path: File path to load from.
        expected_hash: If provided, verify against this hash.
        verify: If True (default), require hash verification.

    Returns:
        Deserialized object.

    Raises:
        ModelIntegrityError: If hash verification fails.
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    if verify:
        resolved_hash = expected_hash
        if resolved_hash is None:
            resolved_hash = _read_hash_file(path)

        if resolved_hash is None:
            raise ModelIntegrityError(
                f"No hash available for {path}. "
                f"Run safe_pickle_dump() to create one, or set verify=False."
            )

        verify_file_hash(path, resolved_hash)
        logger.debug(f"Hash verified for {path}")
    else:
        logger.warning(f"Loading {path} WITHOUT hash verification — unsafe!")

    with open(path, "rb") as f:
        return pickle.load(f)


# ── Migration Helpers ─────────────────────────────────────────────────────


def add_hash_to_existing_file(path: Union[str, Path]) -> str:
    """Compute and write a hash for an existing model file.

    Use this to migrate files saved with raw joblib.dump() / pickle.dump()
    to the safe serialization format.

    Args:
        path: Path to an existing model file.

    Returns:
        SHA-256 hex digest.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    file_hash = compute_file_hash(path)
    _write_hash_file(path, file_hash)
    logger.info(f"Added hash to existing file {path}: {file_hash[:16]}...")
    return file_hash


def has_hash(path: Union[str, Path]) -> bool:
    """Check if a file has a companion .hash file."""
    return _hash_path(Path(path)).exists()


def require_hash(path: Union[str, Path]) -> None:
    """Ensure a file has a companion .hash file. Add one if missing."""
    if not has_hash(path):
        add_hash_to_existing_file(path)


__all__ = [
    "safe_joblib_dump",
    "safe_joblib_load",
    "safe_pickle_dump",
    "safe_pickle_load",
    "compute_file_hash",
    "verify_file_hash",
    "add_hash_to_existing_file",
    "has_hash",
    "require_hash",
    "ModelIntegrityError",
    "ModelNotFoundError",
]
