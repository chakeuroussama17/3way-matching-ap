"""
Explainability layer — turns every model decision into a short, signed list
of the features that drove it.

Uses SHAP where it is reliable:
- ``shap.TreeExplainer`` for the tree models (IsolationForest, XGBoost)
- ``shap.LinearExplainer`` for logistic-regression vendor risk

and degrades gracefully when SHAP is unavailable or errors:
- tree models fall back to ``feature_importances_`` weighted by the feature's
  standardised deviation,
- linear models fall back to ``coef * standardised_value`` (which *is* the
  exact SHAP contribution for a linear model),
- the anomaly model falls back to per-feature z-scores (how unusual each
  input is versus the training mean) — always interpretable, never wrong.

The public functions return a uniform shape::

    {
      "method": "tree_shap" | "linear_shap" | "importance_fallback" | ...,
      "top_features": [
        {"feature": "amount", "value": 51200.0,
         "contribution": 0.61, "direction": "increases"},
        ...
      ]
    }
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ml.base import get_logger

log = get_logger("explainability")

try:  # pragma: no cover - availability depends on environment
    import shap  # type: ignore

    _SHAP_AVAILABLE = True
except Exception:  # noqa: BLE001
    shap = None  # type: ignore
    _SHAP_AVAILABLE = False


def _format(
    feature_names: list[str],
    raw_values: dict[str, Any],
    contributions: np.ndarray,
    method: str,
    top_k: int,
) -> dict[str, Any]:
    """Sort, trim and label contributions into the public output shape."""
    order = np.argsort(-np.abs(contributions))[:top_k]
    top = []
    for idx in order:
        name = feature_names[idx]
        contrib = float(contributions[idx])
        top.append(
            {
                "feature": name,
                "value": raw_values.get(name),
                "contribution": round(contrib, 4),
                "direction": "increases" if contrib >= 0 else "decreases",
            }
        )
    return {"method": method, "top_features": top}


# ── Anomaly (IsolationForest) ─────────────────────────────────────────────────
def explain_anomaly(detector, features: dict, top_k: int = 5) -> dict[str, Any]:
    """Explain why an invoice received its anomaly score.

    Contributions are oriented so a **positive** value means the feature made
    the invoice look *more* anomalous.
    """
    from ml.anomaly_detection import FEATURES

    x_raw = np.array([[float(features.get(f, 0.0) or 0.0) for f in FEATURES]], dtype=float)
    x_scaled = detector.scaler.transform(x_raw)

    if _SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(detector.model)
            sv = explainer.shap_values(x_scaled)
            sv = np.asarray(sv).reshape(-1)
            # score_samples is higher-when-normal, so negate to point at anomaly.
            return _format(FEATURES, features, -sv, "tree_shap", top_k)
        except Exception as exc:  # noqa: BLE001
            log.warning("SHAP anomaly explanation failed (%s); using z-score fallback.", exc)

    # Fallback: standardised deviation is exactly x_scaled; larger |z| == more unusual.
    return _format(FEATURES, features, x_scaled.reshape(-1), "zscore_fallback", top_k)


# ── Vendor risk (logistic regression) ─────────────────────────────────────────
def explain_vendor_risk(model_wrapper, features: dict, top_k: int = 5) -> dict[str, Any]:
    """Explain a vendor's risk score via signed log-odds contributions."""
    feature_names = model_wrapper.FEATURES
    x_raw = np.array([[float(features.get(f, 0.0) or 0.0) for f in feature_names]], dtype=float)
    x_scaled = model_wrapper.scaler.transform(x_raw)

    clf = model_wrapper.model
    if _SHAP_AVAILABLE:
        try:
            explainer = shap.LinearExplainer(clf, model_wrapper.background_)
            sv = np.asarray(explainer.shap_values(x_scaled)).reshape(-1)
            return _format(feature_names, features, sv, "linear_shap", top_k)
        except Exception as exc:  # noqa: BLE001
            log.warning("SHAP linear explanation failed (%s); using coef fallback.", exc)

    # Exact linear contribution == coef * standardised value.
    contrib = (clf.coef_.reshape(-1)) * x_scaled.reshape(-1)
    return _format(feature_names, features, contrib, "linear_coef_fallback", top_k)


# ── Workflow (XGBoost) ────────────────────────────────────────────────────────
def explain_workflow(model_wrapper, features: dict, target: str = "rejection", top_k: int = 5) -> dict[str, Any]:
    """Explain a rejection/escalation prediction for one invoice.

    ``target`` selects which sub-model to explain ("rejection" or "escalation").
    """
    feature_names = model_wrapper.FEATURES
    estimator = model_wrapper.estimator_for(target)
    x_raw = np.array([[float(features.get(f, 0.0) or 0.0) for f in feature_names]], dtype=float)

    if _SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(estimator)
            sv = explainer.shap_values(x_raw)
            # Binary XGBClassifier returns a single array; multiclass a list.
            if isinstance(sv, list):
                sv = sv[-1]
            sv = np.asarray(sv).reshape(-1)
            return _format(feature_names, features, sv, "tree_shap", top_k)
        except Exception as exc:  # noqa: BLE001
            log.warning("SHAP workflow explanation failed (%s); using importance fallback.", exc)

    # Fallback: importance weighted by signed deviation from the feature mean.
    importances = getattr(estimator, "feature_importances_", np.ones(len(feature_names)))
    means = np.asarray(model_wrapper.feature_means_, dtype=float)
    deviation = x_raw.reshape(-1) - means
    contrib = importances * np.sign(deviation) * np.abs(deviation) / (np.abs(means) + 1e-9)
    return _format(feature_names, features, contrib, "importance_fallback", top_k)


def shap_available() -> bool:
    """Whether the full SHAP backend is installed (vs. the fallbacks)."""
    return _SHAP_AVAILABLE
