"""
Approval-workflow prediction with gradient-boosted trees (XGBoost).

Three heads share one feature vector:
- ``rejection`` classifier  → P(invoice is rejected)
- ``escalation`` classifier → P(invoice needs escalation)
- ``approval_time`` regressor → estimated approval time (hours)

The predictions drive **auto-routing**: low-risk invoices below the
auto-approve threshold are recommended for straight-through processing;
everything else is routed to manual review. All heads are explainable via
``shap.TreeExplainer`` (see ``explainability``).

If ``xgboost`` is unavailable the model transparently falls back to
scikit-learn's ``HistGradientBoosting*`` so the pipeline still trains and
serves.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from ml.base import ModelMetadata, get_logger, load_artifact, new_version, save_artifact

log = get_logger("workflow_prediction")

# anomaly_score and risk_score are produced by the other models; the trainer
# enriches the dataframe with them before fitting this model.
FEATURES: list[str] = [
    "amount",
    "line_item_count",
    "vendor_invoice_count_30d",
    "unit_price_deviation_pct",
    "quantity_mismatch",   # from the rule engine's quantity check
    "is_duplicate",        # from duplicate detection
    "po_match_quality",
    "anomaly_score",
    "risk_score",
]

# Auto-approval gate: an invoice is recommended for auto-approval only when
# BOTH probabilities sit below these thresholds.
AUTO_APPROVE_MAX_REJECTION = 0.15
AUTO_APPROVE_MAX_ESCALATION = 0.25

try:  # pragma: no cover
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore

    _XGB_AVAILABLE = True
except Exception:  # noqa: BLE001
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    _XGB_AVAILABLE = False


def _make_classifier():
    if _XGB_AVAILABLE:
        return XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, random_state=42)


def _make_regressor():
    if _XGB_AVAILABLE:
        return XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9, random_state=42, n_jobs=-1
        )
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.1, random_state=42)


class WorkflowPredictor:
    """XGBoost approval-workflow predictor with auto-routing recommendations."""

    NAME = "workflow_prediction"
    FEATURES = FEATURES

    def __init__(
        self,
        rejection_clf=None,
        escalation_clf=None,
        approval_reg=None,
        feature_means: Optional[list[float]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.rejection_clf = rejection_clf
        self.escalation_clf = escalation_clf
        self.approval_reg = approval_reg
        self.feature_means_ = feature_means or [0.0] * len(FEATURES)
        self.metadata = metadata or {}

    @property
    def is_trained(self) -> bool:
        return self.rejection_clf is not None and self.escalation_clf is not None

    def estimator_for(self, target: str):
        """Return the sub-estimator for a SHAP explanation of ``target``."""
        return {"rejection": self.rejection_clf, "escalation": self.escalation_clf}.get(target, self.rejection_clf)

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> ModelMetadata:
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(
                f"Training data missing required columns: {missing}. "
                "Did you enrich it with anomaly_score / risk_score first?"
            )

        X = df[FEATURES].astype(float).fillna(0.0).to_numpy()
        y_reject = df.get("rejected", pd.Series([0] * len(df))).astype(int).to_numpy()
        y_escal = df.get("escalated", pd.Series([0] * len(df))).astype(int).to_numpy()
        y_time = df.get("days_to_approval", pd.Series([0.0] * len(df))).astype(float).to_numpy()

        self.feature_means_ = X.mean(axis=0).tolist()

        # Honest cross-validated metrics (reported), then fit final models on all data.
        metrics: dict[str, float] = self._cv_metrics(X, y_reject, y_escal, y_time)

        self.rejection_clf = _make_classifier().fit(X, y_reject)
        self.escalation_clf = _make_classifier().fit(X, y_escal)
        self.approval_reg = _make_regressor().fit(X, y_time)

        meta = ModelMetadata(
            name=self.NAME,
            version=new_version(),
            trained_at=new_version(),
            n_samples=len(df),
            feature_names=FEATURES,
            metrics=metrics,
            extra={"backend": "xgboost" if _XGB_AVAILABLE else "sklearn_histgb"},
        )
        self.metadata = meta.to_dict()
        return meta

    def _cv_metrics(self, X, y_reject, y_escal, y_time) -> dict[str, float]:
        """5-fold cross-validated metrics — stable and honest on small data."""
        metrics: dict[str, float] = {}
        if len(X) < 50:
            return metrics
        cv = 5
        for name, y in (("rejection", y_reject), ("escalation", y_escal)):
            # Need at least `cv` positives and negatives for stratified folds.
            if min(int(y.sum()), int((1 - y).sum())) >= cv:
                aucs = cross_val_score(_make_classifier(), X, y, cv=cv, scoring="roc_auc")
                metrics[f"{name}_cv_auc"] = round(float(aucs.mean()), 4)
        maes = -cross_val_score(_make_regressor(), X, y_time, cv=cv, scoring="neg_mean_absolute_error")
        metrics["approval_time_cv_mae_days"] = round(float(maes.mean()), 3)
        return metrics

    # ── Prediction ────────────────────────────────────────────────────────────
    def _vectorize(self, features: dict) -> np.ndarray:
        return np.array([[float(features.get(f, 0.0) or 0.0) for f in FEATURES]], dtype=float)

    def predict(self, features: dict) -> dict[str, Any]:
        """Predict rejection / escalation probability, approval time and routing."""
        if not self.is_trained:
            raise RuntimeError("WorkflowPredictor is not trained. Run model_trainer first.")

        X = self._vectorize(features)
        p_reject = float(self.rejection_clf.predict_proba(X)[0, 1])
        p_escal = float(self.escalation_clf.predict_proba(X)[0, 1])
        est_days = float(max(0.0, self.approval_reg.predict(X)[0]))

        auto_approve = p_reject <= AUTO_APPROVE_MAX_REJECTION and p_escal <= AUTO_APPROVE_MAX_ESCALATION
        if auto_approve:
            route = "auto_approve"
        elif p_reject >= 0.5:
            route = "manual_review_high_risk"
        else:
            route = "manual_review"

        return {
            "rejection_probability": round(p_reject, 4),
            "escalation_probability": round(p_escal, 4),
            "estimated_approval_time_hours": round(est_days * 24, 1),
            "routing_recommendation": route,
            "auto_approve": auto_approve,
            "features": {f: features.get(f, 0.0) for f in FEATURES},
        }

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained WorkflowPredictor.")
        meta = ModelMetadata(**{**_default_meta(self.NAME), **self.metadata})
        save_artifact(
            self.NAME,
            {
                "rejection_clf": self.rejection_clf,
                "escalation_clf": self.escalation_clf,
                "approval_reg": self.approval_reg,
                "feature_means": self.feature_means_,
            },
            meta,
        )

    @classmethod
    def load(cls) -> Optional["WorkflowPredictor"]:
        bundle = load_artifact(cls.NAME)
        if not bundle:
            return None
        p = bundle["payload"]
        return cls(
            rejection_clf=p["rejection_clf"],
            escalation_clf=p["escalation_clf"],
            approval_reg=p["approval_reg"],
            feature_means=p.get("feature_means"),
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
_MODEL: Optional[WorkflowPredictor] = None


def get_workflow_model(reload: bool = False) -> Optional[WorkflowPredictor]:
    global _MODEL
    if _MODEL is None or reload:
        _MODEL = WorkflowPredictor.load()
        if _MODEL is None:
            log.warning("Workflow model not trained yet; /predict-workflow will 503.")
    return _MODEL
