from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from database import get_db, MatchCase
from extractor import extract_document
from matcher import run_matching
from models import (
    AnomalyRequest,
    ExplainRequest,
    TrainRequest,
    VendorRiskRequest,
    WorkflowRequest,
)

from ml import explainability, monitoring
from ml.anomaly_detection import get_anomaly_model
from ml.base import get_logger
from ml.smart_matching import get_smart_matcher
from ml.vendor_risk import get_vendor_risk_model
from ml.workflow_prediction import get_workflow_model
import ml_integration

log = get_logger("api")

app = FastAPI(
    title="3-Way Matching AP Automation API",
    description="Rule-based 3-way matching enhanced with ML: anomaly detection, "
                "explainability, smart matching, vendor risk and workflow prediction.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm_models() -> None:
    """Preload model singletons; auto-train on synthetic data if none exist.

    Auto-training makes a fresh deployment (e.g. Render, where the filesystem is
    ephemeral) self-sufficient: the ML endpoints work on first boot without a
    manual training step. It is best-effort — a failure never blocks startup,
    it just leaves the ML endpoints returning 503 until models are trained.
    """
    status = ml_integration.ml_status()
    untrained = [k for k, v in status["models"].items() if not v["trained"]]
    if untrained:
        log.info("Untrained models %s — auto-training on synthetic data...", untrained)
        try:
            from ml.model_trainer import train_all
            train_all()
            ml_integration.reload_all_models()
            status = ml_integration.ml_status()
        except Exception as exc:  # noqa: BLE001
            log.warning("Startup auto-training failed (ML endpoints will 503): %s", exc)
    trained = [k for k, v in status["models"].items() if v["trained"]]
    log.info("Models ready on startup: %s | capabilities=%s", trained, status["capabilities"])


@app.get("/", response_class=HTMLResponse)
def root():
    """Simple HTML landing page linking to docs, cases and the ML endpoints."""
    return """
    <html>
        <head><title>MindHive · AP Intelligence API</title></head>
        <body style="font-family: sans-serif; padding: 40px; background: #0E1B2C; color: #DCE5F0; max-width: 680px; margin: auto;">
            <h1>⬢ MindHive · AP Intelligence API</h1>
            <p style="color: #8FA1B8;">Accounts Payable automation — extracts PO, GRN and Invoice fields with GPT-4o Vision,
            runs a rule engine, then enriches with ML: anomaly detection, vendor risk, workflow prediction and SHAP explanations.</p>
            <hr style="border-color: #243851; margin: 24px 0;">
            <h3>🔗 Links</h3>
            <p><a href="/docs" style="color: #D8A83E; text-decoration: none;">📄 Interactive API Docs (Swagger UI)</a></p>
            <p><a href="/redoc" style="color: #D8A83E; text-decoration: none;">📘 ReDoc API Docs</a></p>
            <p><a href="/cases" style="color: #D8A83E; text-decoration: none;">📊 All matching cases (JSON)</a></p>
            <p><a href="/ml/status" style="color: #D8A83E; text-decoration: none;">🤖 ML model status (JSON)</a></p>
            <hr style="border-color: #243851; margin: 24px 0;">
            <h3>📌 Core endpoints</h3>
            <table style="width: 100%; border-collapse: collapse; color: #B9C6D8; font-size: 14px;">
                <tr style="border-bottom: 1px solid #243851;"><td style="padding: 7px;"><code style="color: #D8A83E;">POST</code></td><td><code>/match</code></td><td style="padding:7px;">Upload PO + GRN + Invoice, run matching + ML</td></tr>
                <tr style="border-bottom: 1px solid #243851;"><td style="padding: 7px;"><code style="color: #34D399;">GET</code></td><td><code>/cases</code></td><td style="padding:7px;">List all historical cases</td></tr>
                <tr style="border-bottom: 1px solid #243851;"><td style="padding: 7px;"><code style="color: #2196F3;">PATCH</code></td><td><code>/cases/{id}/approve</code></td><td style="padding:7px;">Manually approve a flagged case</td></tr>
            </table>
            <h3 style="margin-top:22px;">🤖 ML endpoints</h3>
            <table style="width: 100%; border-collapse: collapse; color: #B9C6D8; font-size: 14px;">
                <tr style="border-bottom: 1px solid #243851;"><td style="padding: 7px;"><code style="color: #D8A83E;">POST</code></td><td><code>/detect-anomaly</code></td><td style="padding:7px;">Anomaly score + SHAP explanation</td></tr>
                <tr style="border-bottom: 1px solid #243851;"><td style="padding: 7px;"><code style="color: #D8A83E;">POST</code></td><td><code>/vendor-risk</code></td><td style="padding:7px;">Vendor risk score (0-100)</td></tr>
                <tr style="border-bottom: 1px solid #243851;"><td style="padding: 7px;"><code style="color: #D8A83E;">POST</code></td><td><code>/predict-workflow</code></td><td style="padding:7px;">Rejection/escalation + auto-routing</td></tr>
                <tr style="border-bottom: 1px solid #243851;"><td style="padding: 7px;"><code style="color: #34D399;">GET</code></td><td><code>/smart-match</code></td><td style="padding:7px;">Hybrid fuzzy match confidence</td></tr>
            </table>
            <hr style="border-color: #243851; margin: 24px 0;">
            <p style="color: #61748C; font-size: 12px;">FastAPI · SQLAlchemy · GPT-4o Vision · scikit-learn · XGBoost · SHAP</p>
        </body>
    </html>
    """


# ══════════════════════════════════════════════════════════════════════════════
#  Core pipeline (existing — now ML-enriched)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/match")
async def match_documents(
    po:      UploadFile = File(...),
    grn:     UploadFile = File(...),
    invoice: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Extract all three documents, run the rule engine, then enrich with ML."""
    po_data  = extract_document(await po.read(),      po.filename,      "purchase_order")
    grn_data = extract_document(await grn.read(),     grn.filename,     "goods_received_note")
    inv_data = extract_document(await invoice.read(), invoice.filename, "invoice")

    # ── Rule engine (unchanged, authoritative) ──
    result = run_matching(po_data, grn_data, inv_data)

    # ── ML enrichment (best-effort; never breaks the rule result) ──
    ml_block = ml_integration.enrich_match(po_data, grn_data, inv_data, result, db)
    result["ml"] = ml_block

    anomaly = (ml_block.get("anomaly") or {}) if ml_block.get("available") else {}
    workflow = (ml_block.get("workflow") or {}) if ml_block.get("available") else {}

    case = MatchCase(
        po_number    = po_data.get("doc_number"),
        vendor       = po_data.get("vendor_name"),
        po_data      = po_data,
        grn_data     = grn_data,
        invoice_data = inv_data,
        match_result = result,
        status       = "approved" if result["auto_approved"] else "pending",
        flags        = len(result["flags"]),
        anomaly_score         = anomaly.get("anomaly_score"),
        anomaly_band          = anomaly.get("band"),
        risk_score            = (ml_block.get("risk") or {}).get("risk_score"),
        rejection_probability = workflow.get("rejection_probability"),
    )
    db.add(case)
    db.commit()
    db.refresh(case)

    return {
        "case_id":  case.id,
        "po_data":  po_data,
        "grn_data": grn_data,
        "inv_data": inv_data,
        "result":   result,
    }


@app.get("/cases")
def get_cases(db: Session = Depends(get_db)):
    cases = db.query(MatchCase).order_by(MatchCase.created_at.desc()).all()
    return [
        {
            "id":            c.id,
            "created_at":    str(c.created_at),
            "po_number":     c.po_number,
            "vendor":        c.vendor,
            "status":        c.status,
            "flags":         c.flags,
            "auto_approved": c.match_result.get("auto_approved"),
            "confidence":    c.match_result.get("confidence"),
            "anomaly_score": c.anomaly_score,
            "anomaly_band":  c.anomaly_band,
            "risk_score":    c.risk_score,
            "rejection_probability": c.rejection_probability,
        }
        for c in cases
    ]


@app.patch("/cases/{case_id}/approve")
def approve_case(case_id: int, db: Session = Depends(get_db)):
    case = db.query(MatchCase).filter(MatchCase.id == case_id).first()
    if not case:
        raise HTTPException(404, "Case not found")
    case.status = "approved"
    db.commit()

    # Learn from the human decision: log the vendor alias + record the actual.
    po_vendor = (case.po_data or {}).get("vendor_name")
    inv_vendor = (case.invoice_data or {}).get("vendor_name")
    if po_vendor and inv_vendor:
        get_smart_matcher().log_approved_match(po_vendor, inv_vendor)
    monitoring.record_actual(case_id, "approved")
    return {"status": "approved"}


@app.patch("/cases/{case_id}/reject")
def reject_case(case_id: int, db: Session = Depends(get_db)):
    case = db.query(MatchCase).filter(MatchCase.id == case_id).first()
    if not case:
        raise HTTPException(404, "Case not found")
    case.status = "rejected"
    db.commit()
    monitoring.record_actual(case_id, "rejected")
    return {"status": "rejected"}


@app.get("/cases/{case_id}")
def get_case_detail(case_id: int, db: Session = Depends(get_db)):
    case = db.query(MatchCase).filter(MatchCase.id == case_id).first()
    if not case:
        raise HTTPException(404, "Case not found")
    return {
        "id":           case.id,
        "po_data":      case.po_data,
        "grn_data":     case.grn_data,
        "invoice_data": case.invoice_data,
        "match_result": case.match_result,
        "status":       case.status,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ML endpoints
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/ml/status")
def ml_status():
    """Which models are trained and which optional backends are active."""
    return ml_integration.ml_status()


@app.post("/detect-anomaly")
def detect_anomaly(req: AnomalyRequest):
    """Score a single invoice for anomalousness (0-1) with a SHAP explanation."""
    model = get_anomaly_model()
    if model is None:
        raise HTTPException(503, "Anomaly model not trained. Run `python -m ml.model_trainer`.")
    features = req.model_dump(exclude={"top_k"})
    result = model.score(features)
    result["explanation"] = explainability.explain_anomaly(model, features, top_k=req.top_k)
    monitoring.record_prediction("anomaly_detection", features, result)
    return result


@app.post("/vendor-risk")
def vendor_risk(req: VendorRiskRequest):
    """Score an arbitrary vendor feature vector → risk score (0-100)."""
    model = get_vendor_risk_model()
    if model is None:
        raise HTTPException(503, "Vendor risk model not trained. Run `python -m ml.model_trainer`.")
    features = req.model_dump(exclude={"top_k"})
    result = model.score(features)
    result["explanation"] = explainability.explain_vendor_risk(model, features, top_k=req.top_k)
    monitoring.record_prediction("vendor_risk", features, result)
    return result


@app.get("/vendors/risk")
def top_risky_vendors(top: int = Query(10, ge=1, le=100)):
    """Return the top-N highest-risk vendors with incident history."""
    model = get_vendor_risk_model()
    if model is None:
        raise HTTPException(503, "Vendor risk model not trained. Run `python -m ml.model_trainer`.")
    return {"top": top, "vendors": model.top_risky(top)}


@app.post("/predict-workflow")
def predict_workflow(req: WorkflowRequest):
    """Predict rejection/escalation probability, approval time and routing."""
    model = get_workflow_model()
    if model is None:
        raise HTTPException(503, "Workflow model not trained. Run `python -m ml.model_trainer`.")
    features = req.model_dump(exclude={"top_k"})
    result = model.predict(features)
    result["explanation"] = explainability.explain_workflow(model, features, "rejection", top_k=req.top_k)
    monitoring.record_prediction("workflow_prediction", features, result)
    return result


@app.get("/smart-match")
def smart_match(
    po_text: str = Query(..., description="text from the PO (e.g. vendor name)"),
    invoice_text: str = Query(..., description="text from the invoice to compare"),
):
    """Hybrid rule + embedding match confidence with a per-component explanation."""
    return get_smart_matcher().match(po_text, invoice_text)


@app.post("/explain")
def explain(req: ExplainRequest):
    """Explain any model's decision for a given feature vector."""
    if req.model == "anomaly":
        model = get_anomaly_model()
        if model is None:
            raise HTTPException(503, "Anomaly model not trained.")
        return explainability.explain_anomaly(model, req.features, top_k=req.top_k)
    if req.model == "vendor_risk":
        model = get_vendor_risk_model()
        if model is None:
            raise HTTPException(503, "Vendor risk model not trained.")
        return explainability.explain_vendor_risk(model, req.features, top_k=req.top_k)
    if req.model == "workflow":
        model = get_workflow_model()
        if model is None:
            raise HTTPException(503, "Workflow model not trained.")
        return explainability.explain_workflow(model, req.features, req.target, top_k=req.top_k)
    raise HTTPException(400, f"Unknown model '{req.model}'")


@app.get("/ml/monitoring")
def ml_monitoring():
    """Prediction volume + PSI-based feature-drift status per model."""
    return monitoring.monitoring_summary()


@app.post("/train")
def train(req: TrainRequest):
    """Retrain all models from historical data and hot-reload them.

    Synchronous by design so the response carries the training summary — the
    run takes only a few seconds on the demo dataset.
    """
    from ml.model_trainer import train_all

    try:
        summary = train_all(data_path=req.data_path, n_rows=req.rows)
        ml_integration.reload_all_models()
        return {"status": "trained", "summary": summary}
    except Exception as exc:  # noqa: BLE001
        log.error("Training failed: %s", exc)
        raise HTTPException(500, f"Training failed: {exc}")
