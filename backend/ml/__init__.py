"""
ML package for the 3-Way Matching AP Automation pipeline.

This package layers machine-learning models on top of the existing
rule-based engine (``matcher.py``) — it enhances, never replaces it.

Sub-modules
-----------
- ``base``                : shared persistence / versioning / logging helpers
- ``data_generator``      : synthetic historical invoice data for demos
- ``anomaly_detection``   : IsolationForest invoice anomaly scoring
- ``smart_matching``      : embedding + rule hybrid fuzzy matching
- ``explainability``      : SHAP wrapper (with graceful fallback)
- ``vendor_risk``         : logistic-regression vendor risk scoring
- ``workflow_prediction`` : XGBoost approval-workflow prediction
- ``monitoring``          : prediction/actual logging + drift detection
- ``model_trainer``       : one-shot training entry point for all models

Every model degrades gracefully: if an optional heavy dependency
(``shap``, ``xgboost``, ``sentence-transformers``) is not installed, the
relevant feature falls back to a simpler, always-available implementation
instead of crashing the API.
"""

__all__ = [
    "base",
    "data_generator",
    "anomaly_detection",
    "smart_matching",
    "explainability",
    "vendor_risk",
    "workflow_prediction",
    "monitoring",
    "model_trainer",
]
