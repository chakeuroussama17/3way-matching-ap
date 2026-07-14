"""
Invoice anomaly detection with an IsolationForest.

Flags unusual invoices — price outliers, oversized totals, suspicious vendor
cadence, duplicate patterns — as an interpretable **anomaly score in [0, 1]**
plus a red / yellow / green band for the dashboard.

The raw IsolationForest signal (``-score_samples``) is min-max normalised
against the training distribution so the exposed score is stable and
comparable across invoices. Scoring a single invoice is a scaler transform
plus one forest traversal — comfortably under the 500 ms budget.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ml.base import ModelMetadata, get_logger, load_artifact, new_version, save_artifact

log = get_logger("anomaly_detection")

# Feature order is contractual: it must match what training, scoring and the
# SHAP explainer all use. Keep this list as the single source of truth.
FEATURES: list[str] = [
    "amount",
    "line_item_count",
    "vendor_invoice_count_30d",
    "days_to_approval",
    "unit_price_deviation_pct",
    "po_match_quality",
    "is_duplicate",
]

# Band thresholds on the normalised [0,1] anomaly score.
YELLOW_THRESHOLD = 0.40
RED_THRESHOLD = 0.70


def _band(score: float) -> str:
    """Map a normalised anomaly score to a traffic-light band."""
    if score >= RED_THRESHOLD:
        return "red"
    if score >= YELLOW_THRESHOLD:
        return "yellow"
    return "green"


def features_from_invoice(invoice: dict, match_result: Optional[dict] = None) -> dict[str, float]:
    """Derive the anomaly feature vector from an extracted invoice + match result.

    Best-effort: any field the extractor did not populate falls back to a
    neutral default so scoring never raises on partial data.
    """
    match_result = match_result or {}
    line_items = invoice.get("line_items") or []

    def _f(val: Any, default: float = 0.0) -> float:
        try:
            return float(str(val).replace(",", "").replace("$", "").strip())
        except (TypeError, ValueError):
            return default

    return {
        "amount": _f(invoice.get("total_amount")),
        "line_item_count": float(len(line_items)),
        # Cadence isn't known from a single doc; caller may override.
        "vendor_invoice_count_30d": _f(invoice.get("vendor_invoice_count_30d"), 1.0),
        "days_to_approval": _f(invoice.get("days_to_approval"), 5.0),
        "unit_price_deviation_pct": _f(invoice.get("unit_price_deviation_pct"), 0.0),
        "po_match_quality": _f(match_result.get("confidence"), 1.0),
        "is_duplicate": 1.0 if invoice.get("is_duplicate") else 0.0,
    }


class AnomalyDetector:
    """IsolationForest wrapper producing normalised anomaly scores."""

    NAME = "anomaly_detection"

    def __init__(
        self,
        model: Optional[IsolationForest] = None,
        scaler: Optional[StandardScaler] = None,
        score_min: float = 0.0,
        score_max: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.score_min = score_min
        self.score_max = score_max
        self.metadata = metadata or {}

    @property
    def is_trained(self) -> bool:
        return self.model is not None and self.scaler is not None

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame, contamination: float = 0.08, random_state: int = 42) -> ModelMetadata:
        """Fit the IsolationForest on historical invoices.

        Parameters
        ----------
        df
            Historical data containing the columns in :data:`FEATURES`.
        contamination
            Expected proportion of anomalies; passed to IsolationForest.
        """
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Training data missing required columns: {missing}")

        X = df[FEATURES].astype(float).fillna(0.0).to_numpy()

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        ).fit(Xs)

        # Calibrate the [0,1] normalisation against the training distribution.
        raw = -self.model.score_samples(Xs)  # higher == more anomalous
        self.score_min = float(raw.min())
        self.score_max = float(raw.max())

        flagged = int((self._normalise(raw) >= YELLOW_THRESHOLD).sum())
        meta = ModelMetadata(
            name=self.NAME,
            version=new_version(),
            trained_at=new_version(),
            n_samples=len(df),
            feature_names=FEATURES,
            metrics={
                "contamination": contamination,
                "flagged_fraction": round(flagged / max(len(df), 1), 4),
            },
        )
        self.metadata = meta.to_dict()
        return meta

    # ── Scoring ───────────────────────────────────────────────────────────────
    def _normalise(self, raw: np.ndarray) -> np.ndarray:
        span = self.score_max - self.score_min
        if span <= 1e-12:
            return np.clip(raw - self.score_min, 0.0, 1.0)
        return np.clip((raw - self.score_min) / span, 0.0, 1.0)

    def _vectorize(self, features: dict) -> np.ndarray:
        return np.array([[float(features.get(f, 0.0) or 0.0) for f in FEATURES]], dtype=float)

    def score(self, features: dict) -> dict[str, Any]:
        """Score a single invoice.

        Returns
        -------
        dict
            ``anomaly_score`` (0-1), ``band`` (red/yellow/green),
            ``is_anomaly`` (bool) and the ``features`` used (for explainability).
        """
        if not self.is_trained:
            raise RuntimeError("AnomalyDetector is not trained. Run model_trainer first.")

        X = self._vectorize(features)
        Xs = self.scaler.transform(X)
        raw = -self.model.score_samples(Xs)
        score = float(self._normalise(raw)[0])
        band = _band(score)

        return {
            "anomaly_score": round(score, 4),
            "band": band,
            "is_anomaly": band != "green",
            "raw_score": round(float(raw[0]), 4),
            "features": {f: features.get(f, 0.0) for f in FEATURES},
        }

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained AnomalyDetector.")
        meta = ModelMetadata(**{**_default_meta(self.NAME), **self.metadata})
        save_artifact(
            self.NAME,
            {
                "model": self.model,
                "scaler": self.scaler,
                "score_min": self.score_min,
                "score_max": self.score_max,
            },
            meta,
        )

    @classmethod
    def load(cls) -> Optional["AnomalyDetector"]:
        bundle = load_artifact(cls.NAME)
        if not bundle:
            return None
        p = bundle["payload"]
        return cls(
            model=p["model"],
            scaler=p["scaler"],
            score_min=p.get("score_min", 0.0),
            score_max=p.get("score_max", 1.0),
            metadata=bundle.get("metadata", {}),
        )


def _default_meta(name: str) -> dict:
    return {
        "name": name,
        "version": new_version(),
        "trained_at": new_version(),
        "n_samples": 0,
        "feature_names": FEATURES,
        "metrics": {},
        "extra": {},
    }


# ── Module singleton (loaded once on API startup) ─────────────────────────────
_MODEL: Optional[AnomalyDetector] = None


def get_anomaly_model(reload: bool = False) -> Optional[AnomalyDetector]:
    """Return the process-wide AnomalyDetector, loading it from disk on first use."""
    global _MODEL
    if _MODEL is None or reload:
        _MODEL = AnomalyDetector.load()
        if _MODEL is None:
            log.warning("Anomaly model not trained yet; /detect-anomaly will 503.")
    return _MODEL
