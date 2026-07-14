"""
Vendor risk scoring with logistic regression.

A vendor is "risky" when its invoices tend to fail matching or require
escalation. We fit an interpretable logistic model at the invoice level::

    P(failed | mismatch_rate, payment_delay_days, compliance_flags)

and expose:
- :meth:`VendorRiskModel.score` — risk score (0-100) for an arbitrary
  feature vector, and
- **per-vendor profiles** aggregated at training time so the dashboard can
  render a top-risk vendor list with incident history without any live data.

Logistic regression is chosen deliberately: coefficients are directly
interpretable and the SHAP contributions are exact (see ``explainability``).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from ml.base import ModelMetadata, get_logger, load_artifact, new_version, save_artifact

log = get_logger("vendor_risk")

FEATURES: list[str] = ["mismatch_rate", "payment_delay_days", "compliance_flags"]

YELLOW_THRESHOLD = 40.0  # risk score 0-100
RED_THRESHOLD = 70.0


def _band(score: float) -> str:
    if score >= RED_THRESHOLD:
        return "high"
    if score >= YELLOW_THRESHOLD:
        return "medium"
    return "low"


class VendorRiskModel:
    """Logistic-regression vendor risk scorer with cached vendor profiles."""

    NAME = "vendor_risk"
    FEATURES = FEATURES

    def __init__(
        self,
        model: Optional[LogisticRegression] = None,
        scaler: Optional[StandardScaler] = None,
        background: Optional[np.ndarray] = None,
        vendor_profiles: Optional[dict[str, dict]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.background_ = background  # scaled sample for shap.LinearExplainer
        self.vendor_profiles = vendor_profiles or {}
        self.metadata = metadata or {}

    @property
    def is_trained(self) -> bool:
        return self.model is not None and self.scaler is not None

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame, random_state: int = 42) -> ModelMetadata:
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Training data missing required columns: {missing}")

        X = df[FEATURES].astype(float).fillna(0.0).to_numpy()
        # "Risky" == the invoice failed matching (rejected). Escalations are an
        # amount/authority signal handled by the workflow model, not a vendor-
        # quality signal, so they are deliberately excluded here.
        y = df.get("rejected", pd.Series([0] * len(df))).astype(int).to_numpy()

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        # Honest cross-validated metric, then fit the final model on all data.
        metrics: dict[str, float] = {"positive_rate": round(float(y.mean()), 4)}
        if len(np.unique(y)) > 1 and min(int(y.sum()), int((1 - y).sum())) >= 5:
            estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
            aucs = cross_val_score(estimator, Xs, y, cv=5, scoring="roc_auc")
            metrics["cv_auc"] = round(float(aucs.mean()), 4)

        self.model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state).fit(Xs, y)
        self.background_ = Xs[np.random.default_rng(random_state).choice(len(Xs), size=min(100, len(Xs)), replace=False)]
        if len(np.unique(y)) > 1:
            metrics["train_auc"] = round(float(roc_auc_score(y, self.model.predict_proba(Xs)[:, 1])), 4)

        self._build_vendor_profiles(df)

        meta = ModelMetadata(
            name=self.NAME,
            version=new_version(),
            trained_at=new_version(),
            n_samples=len(df),
            feature_names=FEATURES,
            metrics=metrics,
            extra={"n_vendors": len(self.vendor_profiles)},
        )
        self.metadata = meta.to_dict()
        return meta

    def _build_vendor_profiles(self, df: pd.DataFrame) -> None:
        """Aggregate each vendor's mean features + incident history and score them."""
        self.vendor_profiles = {}
        if "vendor_name" not in df.columns:
            return
        for vendor, grp in df.groupby("vendor_name"):
            feats = {f: float(grp[f].astype(float).mean()) for f in FEATURES}
            # Incidents == failed matches (rejections), aligned with the model target.
            incidents = int(grp.get("rejected", pd.Series([0] * len(grp))).astype(int).sum())
            score = self._score_vector(feats)
            self.vendor_profiles[str(vendor)] = {
                "vendor_name": str(vendor),
                "risk_score": score,
                "band": _band(score),
                "invoice_count": int(len(grp)),
                "incident_count": incidents,
                "incident_rate": round(incidents / max(len(grp), 1), 3),
                "features": {k: round(v, 3) for k, v in feats.items()},
            }

    # ── Scoring ───────────────────────────────────────────────────────────────
    def _score_vector(self, features: dict) -> float:
        x = np.array([[float(features.get(f, 0.0) or 0.0) for f in FEATURES]], dtype=float)
        xs = self.scaler.transform(x)
        prob = float(self.model.predict_proba(xs)[0, 1])
        return round(prob * 100, 1)

    def score(self, features: dict) -> dict[str, Any]:
        """Score an arbitrary vendor feature vector → risk_score (0-100) + band."""
        if not self.is_trained:
            raise RuntimeError("VendorRiskModel is not trained. Run model_trainer first.")
        score = self._score_vector(features)
        return {
            "risk_score": score,
            "band": _band(score),
            "features": {f: features.get(f, 0.0) for f in FEATURES},
        }

    def score_vendor(self, vendor_name: str) -> Optional[dict]:
        """Return the cached profile for a known vendor (or ``None``)."""
        return self.vendor_profiles.get(vendor_name)

    def top_risky(self, n: int = 10) -> list[dict]:
        """Return the ``n`` highest-risk vendors, most risky first."""
        return sorted(self.vendor_profiles.values(), key=lambda v: -v["risk_score"])[:n]

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained VendorRiskModel.")
        meta = ModelMetadata(**{**_default_meta(self.NAME), **self.metadata})
        save_artifact(
            self.NAME,
            {
                "model": self.model,
                "scaler": self.scaler,
                "background": self.background_,
                "vendor_profiles": self.vendor_profiles,
            },
            meta,
        )

    @classmethod
    def load(cls) -> Optional["VendorRiskModel"]:
        bundle = load_artifact(cls.NAME)
        if not bundle:
            return None
        p = bundle["payload"]
        return cls(
            model=p["model"],
            scaler=p["scaler"],
            background=p.get("background"),
            vendor_profiles=p.get("vendor_profiles", {}),
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


# ── Module singleton ──────────────────────────────────────────────────────────
_MODEL: Optional[VendorRiskModel] = None


def get_vendor_risk_model(reload: bool = False) -> Optional[VendorRiskModel]:
    global _MODEL
    if _MODEL is None or reload:
        _MODEL = VendorRiskModel.load()
        if _MODEL is None:
            log.warning("Vendor risk model not trained yet; risk endpoints will 503.")
    return _MODEL
