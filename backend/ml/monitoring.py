"""
Lightweight model monitoring: prediction/actual logging and feature-drift
detection via the Population Stability Index (PSI).

- Every served prediction is appended to ``ml/logs/predictions.jsonl`` with a
  timestamp, model name/version and the input features.
- When a case's true outcome becomes known (human approve/reject) it is
  appended to ``ml/logs/actuals.jsonl`` so prediction-vs-actual accuracy can be
  audited offline.
- At training time a per-feature **baseline** (decile bin edges) is saved. At
  any point :func:`compute_drift` compares the distribution of recently served
  features against that baseline and reports PSI per feature.

PSI interpretation (industry convention):
    < 0.10  no significant shift
    0.10-0.25  moderate shift — investigate
    > 0.25  major shift — retrain
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ml.base import LOGS_DIR, get_logger

log = get_logger("monitoring")

PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"
ACTUALS_LOG = LOGS_DIR / "actuals.jsonl"
BASELINE_DIR = LOGS_DIR / "baselines"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

PSI_MODERATE = 0.10
PSI_MAJOR = 0.25


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Logging ───────────────────────────────────────────────────────────────────
def record_prediction(
    model_name: str,
    features: dict[str, Any],
    output: dict[str, Any],
    case_id: Optional[int] = None,
    version: Optional[str] = None,
) -> None:
    """Append a served prediction to the predictions log (best-effort)."""
    try:
        rec = {
            "ts": _now(),
            "model": model_name,
            "version": version,
            "case_id": case_id,
            "features": features,
            "output": output,
        }
        with PREDICTIONS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001 — monitoring must never break serving
        log.warning("Failed to record prediction for %s: %s", model_name, exc)


def record_actual(case_id: int, actual: Any, model_name: str = "workflow_prediction") -> None:
    """Append a realised outcome for a case so predictions can be scored later."""
    try:
        rec = {"ts": _now(), "model": model_name, "case_id": case_id, "actual": actual}
        with ACTUALS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to record actual for case %s: %s", case_id, exc)


# ── Baselines ─────────────────────────────────────────────────────────────────
def save_baseline(model_name: str, df: pd.DataFrame, features: list[str], n_bins: int = 10) -> None:
    """Persist decile bin edges + proportions for each feature at training time."""
    baseline: dict[str, Any] = {"created": _now(), "n_bins": n_bins, "features": {}}
    for f in features:
        if f not in df.columns:
            continue
        col = df[f].astype(float).fillna(0.0).to_numpy()
        # Unique quantile edges (guard against constant / low-cardinality columns).
        edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            edges = np.array([col.min() - 1e-6, col.mean(), col.max() + 1e-6])
        counts, _ = np.histogram(col, bins=edges)
        props = counts / max(counts.sum(), 1)
        baseline["features"][f] = {"edges": edges.tolist(), "proportions": props.tolist()}

    path = BASELINE_DIR / f"{model_name}.json"
    path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    log.info("Saved drift baseline for %s (%d features)", model_name, len(baseline["features"]))


def _psi(expected_props: np.ndarray, actual_props: np.ndarray, eps: float = 1e-6) -> float:
    """Population Stability Index between two binned distributions."""
    e = np.clip(expected_props, eps, None)
    a = np.clip(actual_props, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ── Drift ─────────────────────────────────────────────────────────────────────
def compute_drift(model_name: str, window: int = 200) -> dict[str, Any]:
    """Compare recently served features against the saved baseline via PSI.

    Parameters
    ----------
    model_name
        Which model's baseline + prediction stream to analyse.
    window
        Number of most-recent predictions to include.

    Returns
    -------
    dict
        Per-feature PSI, the overall (max) PSI, a severity label and the
        number of samples the drift was computed over.
    """
    baseline_path = BASELINE_DIR / f"{model_name}.json"
    if not baseline_path.exists():
        return {"status": "no_baseline", "model": model_name}

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    recent = _recent_features(model_name, window)
    if recent.empty:
        return {"status": "no_predictions", "model": model_name, "n_samples": 0}

    per_feature: dict[str, float] = {}
    for f, spec in baseline["features"].items():
        if f not in recent.columns:
            continue
        edges = np.array(spec["edges"], dtype=float)
        counts, _ = np.histogram(recent[f].astype(float).fillna(0.0).to_numpy(), bins=edges)
        actual_props = counts / max(counts.sum(), 1)
        per_feature[f] = round(_psi(np.array(spec["proportions"]), actual_props), 4)

    overall = max(per_feature.values()) if per_feature else 0.0
    severity = "major" if overall > PSI_MAJOR else ("moderate" if overall > PSI_MODERATE else "stable")

    return {
        "status": "ok",
        "model": model_name,
        "n_samples": int(len(recent)),
        "overall_psi": round(overall, 4),
        "severity": severity,
        "drift_detected": overall > PSI_MODERATE,
        "per_feature_psi": dict(sorted(per_feature.items(), key=lambda kv: -kv[1])),
    }


def _recent_features(model_name: str, window: int) -> pd.DataFrame:
    """Load features from the last ``window`` predictions for ``model_name``."""
    if not PREDICTIONS_LOG.exists():
        return pd.DataFrame()
    rows = []
    for line in PREDICTIONS_LOG.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            if rec.get("model") == model_name and isinstance(rec.get("features"), dict):
                rows.append(rec["features"])
        except json.JSONDecodeError:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows[-window:])


def monitoring_summary(models: Optional[list[str]] = None) -> dict[str, Any]:
    """Aggregate prediction counts + drift status across models for the dashboard."""
    models = models or ["anomaly_detection", "vendor_risk", "workflow_prediction"]
    n_predictions = 0
    if PREDICTIONS_LOG.exists():
        n_predictions = sum(1 for _ in PREDICTIONS_LOG.open("r", encoding="utf-8"))
    return {
        "total_predictions_logged": n_predictions,
        "drift": {m: compute_drift(m) for m in models},
    }
