"""
Model persistence: save, load, and version trained models.
Uses joblib for serialization and maintains a model registry.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from betting_intel.config import settings
from betting_intel.utils.safe_serialize import (
    safe_joblib_dump, safe_joblib_load,
    add_hash_to_existing_file,
)
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manages model artifacts with versioning and metadata tracking.
    Models are saved with unique version IDs and metadata for reproducibility.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or (settings.output_dir / "models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._registry_file = self.models_dir / "registry.json"
        self._registry = self._load_registry()
        # Lazy hash-backfill: add SHA-256 hashes to any existing model
        # files that lack them, without blocking startup.
        self._backfill_hashes()

    def _load_registry(self) -> dict:
        if self._registry_file.exists():
            try:
                with open(self._registry_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load model registry, starting fresh: %s", e)
        return {"models": {}, "versions": []}

    def _save_registry(self):
        with open(self._registry_file, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def save(
        self,
        model: Any,
        model_name: str,
        feature_cols: list[str],
        metrics: Optional[dict] = None,
        parameters: Optional[dict] = None,
    ) -> str:
        """
        Save a trained model with versioning.

        Args:
            model: Trained model object
            model_name: Name identifier for the model
            feature_cols: Feature columns used during training
            metrics: Optional performance metrics
            parameters: Optional model parameters

        Returns:
            Version string for the saved model
        """
        version = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True)

        # Save model artifact with hash verification
        model_path = model_dir / f"{version}.joblib"
        model_hash = safe_joblib_dump(model, model_path)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "model_hash": model_hash,
            "feature_cols": feature_cols,
            "metrics": metrics or {},
            "parameters": parameters or {},
            "artifact_path": str(model_path),
        }

        metadata_path = model_dir / f"{version}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Update registry
        if model_name not in self._registry["models"]:
            self._registry["models"][model_name] = []
        self._registry["models"][model_name].append(version)
        self._registry["versions"].append(
            {"model_name": model_name, "version": version, "created_at": metadata["created_at"]}
        )
        self._save_registry()

        logger.info("Model saved: model=%s version=%s path=%s features=%d", model_name, version, model_path, len(feature_cols))
        return version

    def load(self, model_name: str, version: Optional[str] = None) -> tuple[Any, dict]:
        """
        Load a trained model and its metadata.

        Args:
            model_name: Name of the model to load
            version: Specific version to load. If None, loads latest.

        Returns:
            Tuple of (model, metadata_dict)
        """
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"No models found for '{model_name}'")

        if version is None:
            # Load latest version
            versions = self._registry["models"].get(model_name, [])
            if not versions:
                raise FileNotFoundError(f"No versions found for '{model_name}'")
            version = versions[-1]

        model_path = model_dir / f"{version}.joblib"
        metadata_path = model_dir / f"{version}_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        model = safe_joblib_load(model_path)
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        logger.info(
            "Model loaded: model=%s version=%s features=%d integrity=hash_verified",
            model_name, version, len(metadata.get("feature_cols", [])),
        )
        return model, metadata

    def list_models(self) -> list[dict]:
        """List all saved models with their versions."""
        models = []
        for model_name, versions in self._registry["models"].items():
            models.append(
                {
                    "model_name": model_name,
                    "versions": versions,
                    "latest_version": versions[-1] if versions else None,
                    "total_versions": len(versions),
                }
            )
        return models

    def delete_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Delete a model artifact."""
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            return False

        if version:
            paths_to_delete = [
                model_dir / f"{version}.joblib",
                model_dir / f"{version}_metadata.json",
            ]
        else:
            import shutil
            shutil.rmtree(model_dir)
            self._registry["models"].pop(model_name, None)
            self._registry["versions"] = [
                v for v in self._registry["versions"] if v["model_name"] != model_name
            ]
            self._save_registry()
            return True

        for p in paths_to_delete:
            if p.exists():
                p.unlink()

        if model_name in self._registry["models"] and version in self._registry["models"][model_name]:
            self._registry["models"][model_name].remove(version)
        self._registry["versions"] = [
            v for v in self._registry["versions"]
            if not (v["model_name"] == model_name and v["version"] == version)
        ]
        self._save_registry()
        return True


    @staticmethod
    def _backfill_hashes() -> None:
        """Backfill SHA-256 hashes for existing .joblib files that lack them.

        Runs once at init time so module-level imports don't trigger
        filesystem scans. Silently skips inaccessible files.
        """
        # settings is already imported at module level — reuse it
        model_dir = settings.output_dir / "models"
        if not model_dir.exists():
            return
        try:
            for joblib_file in model_dir.rglob("*.joblib"):
                try:
                    add_hash_to_existing_file(joblib_file)
                except Exception:
                    pass
        except Exception:
            pass

model_registry = ModelRegistry()
