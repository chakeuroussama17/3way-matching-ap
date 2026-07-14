"""
Unified training entry point for every model in the pipeline.

Run monthly (or on demand) to (re)train all five models from a historical
invoice dataset and persist them to ``backend/ml/artifacts``.

Usage
-----
From the ``backend/`` directory::

    python -m ml.model_trainer                 # uses/generates sample data
    python -m ml.model_trainer --rows 400      # regenerate 400 synthetic rows
    python -m ml.model_trainer --data mine.csv  # train on your own CSV

Training order matters: the workflow model consumes ``anomaly_score`` and
``risk_score`` as features, so the anomaly and vendor-risk models are trained
first and used to enrich the dataframe before the workflow model is fit.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Allow running as a plain script (python ml/model_trainer.py) as well as a
# module (python -m ml.model_trainer) by ensuring backend/ is importable.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ml.anomaly_detection import AnomalyDetector, FEATURES as ANOMALY_FEATURES  # noqa: E402
from ml.base import DATA_DIR, get_logger  # noqa: E402
from ml.data_generator import generate_training_data  # noqa: E402
from ml.monitoring import save_baseline  # noqa: E402
from ml.smart_matching import SmartMatcher  # noqa: E402
from ml.vendor_risk import VendorRiskModel, FEATURES as RISK_FEATURES  # noqa: E402
from ml.workflow_prediction import WorkflowPredictor, FEATURES as WORKFLOW_FEATURES  # noqa: E402

log = get_logger("model_trainer")


def _load_or_generate(data_path: Optional[str], n_rows: int) -> pd.DataFrame:
    """Load the training CSV if present/specified, else generate synthetic data."""
    if data_path:
        log.info("Loading training data from %s", data_path)
        return pd.read_csv(data_path)

    default = DATA_DIR / "sample_training_data.csv"
    if default.exists():
        log.info("Loading existing training data from %s", default)
        return pd.read_csv(default)

    log.info("No training data found; generating %d synthetic rows.", n_rows)
    return generate_training_data(n_rows=n_rows)


def _enrich(df: pd.DataFrame, anomaly: AnomalyDetector, vendor: VendorRiskModel) -> pd.DataFrame:
    """Add anomaly_score and risk_score columns the workflow model needs."""
    df = df.copy()

    # Batch anomaly score (fast: one scaler transform + one forest pass).
    X = df[ANOMALY_FEATURES].astype(float).fillna(0.0).to_numpy()
    raw = -anomaly.model.score_samples(anomaly.scaler.transform(X))
    df["anomaly_score"] = anomaly._normalise(raw)

    # Risk score per row from its vendor-risk features.
    df["risk_score"] = [
        vendor._score_vector({f: row[f] for f in RISK_FEATURES}) for _, row in df.iterrows()
    ]
    return df


def train_all(
    df: Optional[pd.DataFrame] = None,
    data_path: Optional[str] = None,
    n_rows: int = 300,
) -> dict[str, Any]:
    """Train, evaluate and persist all five models.

    Returns a summary dict of ``{model_name: {version, metrics}}``.
    """
    if df is None:
        df = _load_or_generate(data_path, n_rows)
    log.info("Training on %d rows, %d columns.", len(df), df.shape[1])

    summary: dict[str, Any] = {}

    # ── 1. Anomaly detection (unsupervised) ──────────────────────────────────
    anomaly = AnomalyDetector()
    meta = anomaly.train(df)
    anomaly.save()
    save_baseline(anomaly.NAME, df, ANOMALY_FEATURES)
    summary[anomaly.NAME] = {"version": meta.version, "metrics": meta.metrics}

    # ── 2. Vendor risk (logistic regression) ─────────────────────────────────
    vendor = VendorRiskModel()
    meta = vendor.train(df)
    vendor.save()
    save_baseline(vendor.NAME, df, RISK_FEATURES)
    summary[vendor.NAME] = {"version": meta.version, "metrics": meta.metrics}

    # ── 3. Enrich with model-derived features for the workflow model ─────────
    enriched = _enrich(df, anomaly, vendor)

    # ── 4. Workflow prediction (XGBoost) ─────────────────────────────────────
    workflow = WorkflowPredictor()
    meta = workflow.train(enriched)
    workflow.save()
    save_baseline(workflow.NAME, enriched, WORKFLOW_FEATURES)
    summary[workflow.NAME] = {"version": meta.version, "metrics": meta.metrics}

    # ── 5. Smart matcher (alias table seed + approved-log fold-in) ────────────
    matcher = SmartMatcher()
    meta = matcher.train(df)
    matcher.save()
    summary[matcher.NAME] = {"version": meta.version, "metrics": meta.metrics}

    log.info("Training complete for %d models.", len(summary))
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 68)
    print("  MODEL TRAINING SUMMARY")
    print("=" * 68)
    for name, info in summary.items():
        print(f"\n  {name}")
        print(f"    version : {info['version']}")
        for k, v in info["metrics"].items():
            print(f"    {k:<24}: {v}")
    print("\n" + "=" * 68)
    print("  Artifacts saved to backend/ml/artifacts/")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train all AP-automation ML models.")
    parser.add_argument("--data", type=str, default=None, help="path to a historical CSV (defaults to sample data)")
    parser.add_argument("--rows", type=int, default=300, help="synthetic rows to generate if no data exists")
    args = parser.parse_args()

    result = train_all(data_path=args.data, n_rows=args.rows)
    _print_summary(result)
