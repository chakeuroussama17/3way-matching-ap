"""
Shared infrastructure for all ML models: filesystem layout, structured
logging, model persistence and versioning.

Design goals
------------
- **Location-independent paths.** Paths are resolved relative to this file
  so models load correctly whether the API is started from ``backend/`` or
  the repo root.
- **Versioned persistence.** Every ``save`` writes two files: a stable
  ``<name>.joblib`` (the "latest" pointer the API loads on startup) and an
  immutable ``<name>__<version>.joblib`` snapshot for rollback/audit.
- **Metadata travels with the model.** Training sample count, feature names
  and evaluation metrics are pickled alongside the estimator so the API and
  dashboard can display provenance without a separate registry.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib

# ── Filesystem layout ─────────────────────────────────────────────────────────
# base.py lives in  backend/ml/base.py  →  parents[1] == backend/
_BACKEND_DIR = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = _BACKEND_DIR / "ml" / "artifacts"   # saved *.joblib models
DATA_DIR      = _BACKEND_DIR / "data"               # training data / logs source
LOGS_DIR      = _BACKEND_DIR / "ml" / "logs"        # prediction + drift logs

for _d in (ARTIFACTS_DIR, DATA_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Logging ───────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a single stream handler.

    Idempotent: repeated calls for the same ``name`` do not stack handlers.
    """
    logger = logging.getLogger(f"ml.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(os.getenv("ML_LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger


log = get_logger("base")


# ── Model metadata ────────────────────────────────────────────────────────────
@dataclass
class ModelMetadata:
    """Provenance stored next to every trained estimator."""

    name: str
    version: str
    trained_at: str
    n_samples: int
    feature_names: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_version() -> str:
    """UTC timestamp version tag, e.g. ``20260714T091500Z``."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ── Persistence ───────────────────────────────────────────────────────────────
def save_artifact(name: str, payload: dict[str, Any], metadata: ModelMetadata) -> Path:
    """Persist a model bundle to disk under ``ARTIFACTS_DIR``.

    Parameters
    ----------
    name
        Logical model name, e.g. ``"anomaly_detection"``.
    payload
        Arbitrary dict of fitted objects (estimator, scaler, encoders, ...).
    metadata
        :class:`ModelMetadata` describing this training run.

    Returns
    -------
    Path
        Path to the stable ``<name>.joblib`` file the API loads on startup.
    """
    bundle = {"payload": payload, "metadata": metadata.to_dict()}

    stable = ARTIFACTS_DIR / f"{name}.joblib"
    snapshot = ARTIFACTS_DIR / f"{name}__{metadata.version}.joblib"

    joblib.dump(bundle, snapshot)
    joblib.dump(bundle, stable)

    # A tiny human-readable sidecar so provenance is greppable without joblib.
    (ARTIFACTS_DIR / f"{name}.meta.json").write_text(
        json.dumps(metadata.to_dict(), indent=2), encoding="utf-8"
    )

    log.info("Saved model '%s' version %s (%d samples)", name, metadata.version, metadata.n_samples)
    return stable


def load_artifact(name: str) -> Optional[dict[str, Any]]:
    """Load the latest bundle for ``name``.

    Returns
    -------
    dict | None
        ``{"payload": ..., "metadata": ...}`` or ``None`` if no model has
        been trained yet (callers must handle the untrained case).
    """
    stable = ARTIFACTS_DIR / f"{name}.joblib"
    if not stable.exists():
        log.warning("No trained artifact for '%s' at %s", name, stable)
        return None
    try:
        return joblib.load(stable)
    except Exception as exc:  # noqa: BLE001 — never let a bad file crash startup
        log.error("Failed to load artifact '%s': %s", name, exc)
        return None


def artifact_exists(name: str) -> bool:
    return (ARTIFACTS_DIR / f"{name}.joblib").exists()
