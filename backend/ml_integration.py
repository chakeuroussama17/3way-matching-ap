"""
Glue layer between the document pipeline / database and the ML models.

The ``ml`` package is intentionally pure (it never imports the DB). This
module lives at the backend root and bridges the two: it derives model
features from the extracted PO/GRN/Invoice plus the rule-engine result and
recent DB history, runs the models, and records predictions for drift
monitoring.

Everything here is **best-effort**: any failure returns a structured
"unavailable" payload and is logged, but never propagates — the core rule-
based matching must keep working even if a model is missing or a dependency
is absent.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from database import MatchCase
from ml import explainability, monitoring
from ml.anomaly_detection import FEATURES as ANOMALY_FEATURES
from ml.anomaly_detection import features_from_invoice, get_anomaly_model
from ml.base import artifact_exists, get_logger
from ml.smart_matching import get_smart_matcher
from ml.vendor_risk import get_vendor_risk_model
from ml.workflow_prediction import FEATURES as WORKFLOW_FEATURES
from ml.workflow_prediction import get_workflow_model

log = get_logger("integration")

NEUTRAL_RISK = 50.0  # risk score for a vendor with no history


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(str(val).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return default


# ── Feature derivation from documents + history ───────────────────────────────
def unit_price_deviation_pct(po: dict, invoice: dict) -> float:
    """Largest per-line unit-price deviation (%) between PO and invoice."""
    def _index(items):
        out = {}
        for it in items or []:
            desc = str(it.get("description") or "").strip().lower()
            if desc:
                out[desc] = it
        return out

    po_items, inv_items = _index(po.get("line_items")), _index(invoice.get("line_items"))
    worst = 0.0
    for desc, po_item in po_items.items():
        pp = _safe_float(po_item.get("unit_price"))
        ip = _safe_float((inv_items.get(desc) or {}).get("unit_price"))
        if pp and ip:
            worst = max(worst, abs(pp - ip) / pp * 100)
    return round(worst, 3)


def vendor_invoice_count_30d(db: Session, vendor: Optional[str]) -> int:
    """Count how many cases this vendor has in the last 30 days."""
    if not vendor:
        return 1
    cutoff = datetime.utcnow() - timedelta(days=30)
    n = (
        db.query(func.count(MatchCase.id))
        .filter(MatchCase.vendor == vendor, MatchCase.created_at >= cutoff)
        .scalar()
    )
    return int(n or 0) + 1  # +1 to include the invoice being processed


def build_features(po: dict, grn: dict, invoice: dict, result: dict, db: Optional[Session]) -> dict[str, float]:
    """Assemble the union of features every model needs from live documents."""
    vendor = invoice.get("vendor_name") or po.get("vendor_name")
    count_30d = vendor_invoice_count_30d(db, vendor) if db is not None else 1
    dev = unit_price_deviation_pct(po, invoice)

    anomaly_feats = features_from_invoice(invoice, result)
    anomaly_feats["vendor_invoice_count_30d"] = count_30d
    anomaly_feats["unit_price_deviation_pct"] = dev

    # Vendor risk uses a cached profile when the vendor is known.
    risk_model = get_vendor_risk_model()
    profile = risk_model.score_vendor(str(vendor)) if (risk_model and vendor) else None
    risk_score = profile["risk_score"] if profile else NEUTRAL_RISK

    return {
        **anomaly_feats,
        "quantity_mismatch": 0.0 if result.get("quantity_match", True) else 1.0,
        "po_match_quality": _safe_float(result.get("confidence"), 1.0),
        "risk_score": risk_score,
        "_vendor": vendor,
        "_risk_source": "profile" if profile else "neutral_default",
    }


# ── Enrichment used by /match ─────────────────────────────────────────────────
def enrich_match(po: dict, grn: dict, invoice: dict, result: dict, db: Optional[Session]) -> dict[str, Any]:
    """Run anomaly + workflow + risk on a freshly matched invoice.

    Returns an ``ml`` block ready to nest under ``match_result["ml"]``. Never
    raises: on any error the returned block has ``available=False``.
    """
    try:
        feats = build_features(po, grn, invoice, result, db)
        block: dict[str, Any] = {"available": True, "vendor": feats.get("_vendor"), "features": {}}

        # ── Anomaly ──
        anomaly = get_anomaly_model()
        if anomaly:
            a = anomaly.score(feats)
            a["explanation"] = explainability.explain_anomaly(anomaly, feats, top_k=3)
            block["anomaly"] = a
            block["features"].update({f: feats.get(f) for f in ANOMALY_FEATURES})
            monitoring.record_prediction("anomaly_detection", {f: feats.get(f) for f in ANOMALY_FEATURES}, a)

        # ── Workflow ──
        workflow = get_workflow_model()
        if workflow:
            enriched = {**feats, "anomaly_score": block.get("anomaly", {}).get("anomaly_score", 0.0)}
            w = workflow.predict(enriched)
            w["explanation"] = explainability.explain_workflow(workflow, enriched, "rejection", top_k=3)
            block["workflow"] = w
            block["features"].update({f: enriched.get(f) for f in WORKFLOW_FEATURES})
            monitoring.record_prediction("workflow_prediction", {f: enriched.get(f) for f in WORKFLOW_FEATURES}, w)

        # ── Vendor risk (from cached profile) ──
        block["risk"] = {"risk_score": feats.get("risk_score"), "source": feats.get("_risk_source")}
        return block
    except Exception as exc:  # noqa: BLE001
        log.warning("ML enrichment failed (rule result unaffected): %s", exc)
        return {"available": False, "reason": str(exc)}


# ── Status / capability reporting ─────────────────────────────────────────────
def ml_status() -> dict[str, Any]:
    """Report which models are trained + which optional backends are active."""
    def _meta(getter):
        m = getter()
        return m.metadata if m else None

    return {
        "models": {
            "anomaly_detection": {"trained": artifact_exists("anomaly_detection"), "meta": _meta(get_anomaly_model)},
            "vendor_risk": {"trained": artifact_exists("vendor_risk"), "meta": _meta(get_vendor_risk_model)},
            "workflow_prediction": {"trained": artifact_exists("workflow_prediction"), "meta": _meta(get_workflow_model)},
            "smart_matching": {"trained": artifact_exists("smart_matching"), "meta": get_smart_matcher().metadata},
        },
        "capabilities": {
            "shap": explainability.shap_available(),
            "embeddings": get_smart_matcher().uses_embeddings,
        },
    }


def reload_all_models() -> None:
    """Force every model singleton to reload from disk (after retraining)."""
    get_anomaly_model(reload=True)
    get_vendor_risk_model(reload=True)
    get_workflow_model(reload=True)
    get_smart_matcher(reload=True)
